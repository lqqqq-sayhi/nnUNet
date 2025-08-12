# import torch
# from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
# from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# import numpy as np

# class FastNNUNetPredictor(nnUNetPredictor):
#     def __init__(self, device=torch.device('cuda'), use_gaussian=False):
#         super().__init__(
#             tile_step_size=0.5,
#             use_gaussian=use_gaussian,
#             use_mirroring=False,
#             perform_everything_on_device=True,
#             device=device,
#             verbose=False,
#             verbose_preprocessing=False
#         )

#     def predict_from_folder(self, input_folder, output_folder):
#         maybe_mkdir_p(output_folder)
#         from glob import glob
#         png_files = sorted(glob(join(input_folder, '*.png')))
#         for img_path in png_files:
#             img, props = SimpleITKIO().read_images([img_path])
#             iterator = self.get_data_iterator_from_raw_npy_data(img, None, props, None, num_processes=1)
#             for batch in iterator:
#                 data = torch.from_numpy(batch['data'])
#                 logits = self.predict_logits_from_preprocessed_data(data)
#                 seg = logits.argmax(0).cpu().numpy().astype(np.uint8)
#                 out_path = join(output_folder, img_path.split('/')[-1])
#                 SimpleITKIO().write_seg(seg, props, out_path)
"""
Lightweight fast nnU-Net predictor subclass with optional Gaussian blending and PNG support.

This file defines FastNNUNetPredictor which is a thin subclass of the original
nnUNetPredictor that:
 - defaults to single-fold inference,
 - optionally enables/disables gaussian weighting for sliding-window blending,
 - disables TTA (mirroring) by default for speed,
 - includes a convenience `predict_from_folder()` that reads PNG files from a folder without iterator,
   runs preprocessing and inference, then saves segmentation results.
"""

import os
from glob import glob
import time
from typing import Optional

import numpy as np
import torch

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
from nnunetv2.utilities.helpers import empty_cache

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
    """

    def __init__(self,
                 device: torch.device = torch.device('cuda'),
                 use_gaussian: bool = False,
                 use_mirroring: bool = False,
                 tile_step_size: float = 0.5,
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = False):
        # initialize parent with our chosen defaults
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
        # keep track of the Gaussian flag at this object level (parent already stores it)
        self._fast_mode = True

        self.total_start_time = None  # 添加总开始时间
        self.inference_times = []     # 存储每个案例的推理时间

    def predict_from_folder(self,
                            input_folder: str,
                            output_folder: str,
                            file_pattern: str = '*.png',
                            num_processes_preprocessing: int = 1,
                            write_dtype: Optional[np.dtype] = None):
        """
        Predict all files matching `file_pattern` in `input_folder` and write
        segmentations to `output_folder`.

        Parameters
        ----------
        input_folder : str
            Folder containing input images (PNG or any format SimpleITK supports).
        output_folder : str
            Folder where outputs will be saved.
        file_pattern : str
            Glob pattern for input images (default '*.png').
        num_processes_preprocessing : int
            Number of processes used for preprocessing iterator (we use 1 by default).
        write_dtype : numpy dtype, optional
            If provided, cast saved segmentation to this dtype (e.g., np.uint8).
        """

        maybe_mkdir_p(output_folder)
        # glob all matching files
        search = join(input_folder, file_pattern)
        files = sorted(glob(search))
        if len(files) == 0:
            print(f"No files found matching {search}. Nothing to do.")
            return

        io = NaturalImage2DIO()        

        # iterate images one-by-one (single-process preprocessing)
        for img_path in files:
            case_start_time = time.time()  # 记录案例开始时间

            basename = os.path.basename(img_path)
            print(f"\n[fast predict] Processing: {img_path}")

            # read image (SimpleITKIO returns (image_array, properties_dict))
            img, props = io.read_images([img_path])

            # Patch props['spacing'] for 2D PNGs if missing or wrong length
            # remove resampling (PNG is already the working resolution), 1.0 is fine — no scaling applied.
            if 'spacing' not in props or len(props['spacing']) != len(img.shape[1:]):
                # len(img.shape[1:]) = number of spatial dims (2 for PNG)
                props['spacing'] = tuple([1.0] * len(img.shape[1:]))
            
            # Build an iterator that yields a preprocessed batch dict.
            iterator = self.get_data_iterator_from_raw_npy_data(
                img, None, props, None,
                num_processes=num_processes_preprocessing
            )

            # `iterator` yields dicts; for PNG/2D single-case there will be one batch
            for batch in iterator:
                data = batch['data']
                if not isinstance(data, torch.Tensor):
                    data = torch.from_numpy(data)

                # run prediction (returns logits tensor)
                logits = self.predict_logits_from_preprocessed_data(data)
                # convert to segmentation (argmax across heads/channels)
                seg = logits.argmax(0).cpu().numpy()

                # cast dtype if requested
                if write_dtype is not None:
                    seg = seg.astype(write_dtype)
                else:
                    # common default: uint8 if labels small
                    seg = seg.astype(np.uint8)

                out_path = join(output_folder, basename)
                # write segmentation using SimpleITKIO helper (keeps spacing/origin from props)
                io.write_seg(seg, out_path, props)

                print(f"[fast predict] Wrote: {out_path}")

            case_time = time.time() - case_start_time  # 计算总处理时间
        # cleanup caches
        compute_gaussian = getattr(__import__('nnunetv2.inference.sliding_window_prediction', fromlist=['compute_gaussian']), 'compute_gaussian')
        try:
            compute_gaussian.cache_clear()
        except Exception:
            pass
        empty_cache(self.device)
        print("\n[fast predict] Done.")
