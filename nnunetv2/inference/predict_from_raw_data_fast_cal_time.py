"""
Lightweight fast nnU-Net predictor subclass with optional Gaussian blending and PNG support.
Now includes case-level inference time and preprocessing time logging, similar to
predict_from_raw_data_cal_time.py.
"""

import os
from glob import glob
import time
import csv
from typing import Optional, Tuple, Union

import numpy as np
import torch

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.configuration import default_num_processes
from torch._dynamo import OptimizedModule


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
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['case_id', 'inference_time', 'case_time', 'preprocessing_time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for info in self.timing_info:
                writer.writerow(info)
        print(f"[fast predict] Saved timing info to: {csv_path}")

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Same as base method but returns (logits, inference_time).
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
