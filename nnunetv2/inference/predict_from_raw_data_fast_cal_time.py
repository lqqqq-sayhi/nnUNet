"""
Lightweight fast nnU-Net predictor subclass with optional Gaussian blending and PNG support.
Now includes case-level inference time and preprocessing time logging, similar to
predict_from_raw_data_cal_time.py.
"""

from copy import deepcopy
import inspect
import os
from glob import glob
import time
import csv
from typing import Optional, Tuple, Union, List

import numpy as np
import torch

import multiprocessing

from time import sleep
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json

from torch._dynamo import OptimizedModule

from nnunetv2.configuration import default_num_processes

from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy

from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.json_export import recursive_fix_for_json_export

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO




# keep logging minimal
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class FastNNUNetPredictor(nnUNetPredictor):
    """
    Fast variant of nnUNetPredictor:
      - Default: use_gaussian=False, use_mirroring=False (no TTA)
      - perform_everything_on_device=True (if device supports it)
      - Minimal I/O: read images, predict, write segs (no multiprocessing pools)
      - predict_from_folder supports PNG images (and other formats SimpleITK can read)
      - Logs timing info per case to CSV
    """

    def __init__(self,
                 device: torch.device = torch.device('cuda'),
                 use_gaussian: bool = False,
                 use_mirroring: bool = False,
                 tile_step_size: float = 0.5,
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = False):
        super().__init__(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=True,
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose_preprocessing,
            allow_tqdm=allow_tqdm
        )
        self._fast_mode = True
        self.total_start_time = None
        self.inference_times = []
        self.timing_info = []  # store timing info dicts

    def save_timing_info(self, output_dir: str):
        """Save timing info to CSV in output_dir."""
        if not self.timing_info:
            return
        csv_path = os.path.join(output_dir, 'inference_timing.csv')
        write_header = not os.path.exists(csv_path)        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['case_id', 'inference_time', 'case_time', 'preprocessing_time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for info in self.timing_info:
                writer.writerow(info)
        print(f"[fast predict] Saved timing info to: {csv_path}")

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        returns (logits, inference_time).
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None

        start_time = time.time()
        for params in self.list_of_parameters:
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data).to('cpu')
            else:
                prediction += self.predict_sliding_window_return_logits(data).to('cpu')

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        print(f'Inference time: {inference_time:.4f} seconds')

        torch.set_num_threads(n_threads)
        return prediction, inference_time

    def predict_from_folder(self,
                            input_folder: str,
                            output_folder: str,
                            file_pattern: str = '*.png',
                            num_processes_preprocessing: int = 1,
                            write_dtype: Optional[np.dtype] = None):
        maybe_mkdir_p(output_folder)
        files = sorted(glob(join(input_folder, file_pattern)))
        if len(files) == 0:
            print(f"No files found matching {join(input_folder, file_pattern)}. Nothing to do.")
            return

        io = NaturalImage2DIO()        
        self.timing_info.clear()

        for img_path in files:
            case_start_time = time.time()
            basename = os.path.basename(img_path)
            print(f"\n[fast predict] Processing: {img_path}")

            img, props = io.read_images([img_path])
            if 'spacing' not in props or len(props['spacing']) != len(img.shape[1:]):
                props['spacing'] = tuple([1.0] * len(img.shape[1:]))

            iterator = self.get_data_iterator_from_raw_npy_data(
                img, None, props, None,
                num_processes=num_processes_preprocessing
            )

            for batch in iterator:
                if not isinstance(batch['data'], torch.Tensor):
                    data = torch.from_numpy(batch['data'])
                else:
                    data = batch['data']

                logits, inference_time = self.predict_logits_from_preprocessed_data(data)
                seg = logits.argmax(0).cpu().numpy()
                seg = seg.astype(write_dtype or np.uint8)
                io.write_seg(seg, join(output_folder, basename), props)

            case_time = time.time() - case_start_time
            self.timing_info.append({
                'case_id': basename,
                'inference_time': inference_time,
                'case_time': case_time,
                'preprocessing_time': case_time - inference_time
            })

        # cleanup
        try:
            from nnunetv2.inference.sliding_window_prediction import compute_gaussian
            compute_gaussian.cache_clear()
        except Exception:
            pass
        empty_cache(self.device)

        self.save_timing_info(output_folder)
        print("\n[fast predict] Done.")

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        
        # import pdb
        # pdb.set_trace()
        
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        
        result = self.predict_from_data_iterator(
            output_folder=output_folder,
            data_iterator=data_iterator,
            save_probabilities=save_probabilities,
            num_processes_segmentation_export=num_processes_segmentation_export
        )
        
        return result

    def predict_from_data_iterator(self,
                                   output_folder,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """

        self.timing_info.clear()
        case_ids = []

        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                
                case_start_time = time.time()

                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                
                case_id = os.path.basename(preprocessed['ofile']) if preprocessed['ofile'] else f"case_{len(self.timing_info)}"
                case_ids.append(case_id)

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to be swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                # convert to numpy to prevent uncatchable memory alignment errors from multiprocessing serialization of torch tensors
                # prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()
                
                prediction, inference_time = self.predict_logits_from_preprocessed_data(data)
                
                case_time = time.time() - case_start_time

                
                self.timing_info.append({
                    'case_id': case_id,
                    'inference_time': inference_time,
                    'case_time': case_time,
                    'preprocessing_time': case_time - inference_time
                })

                if ofile is not None:
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]


        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)

        self.save_timing_info(output_folder)

        return ret