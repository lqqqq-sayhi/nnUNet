#!/usr/bin/env python3
"""
Notes:
uses FastNNUNetPredictor (from predict_from_raw_data_fast.py), fast nnU-Net predictor 
with optional Gaussian weighting.

CUDA_VISIBLE_DEVICES=1 python3 /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/nnUNet/nnunetv2/inference/infer_without_CLI_fast.py \
-i /mnt/hdd2/task2/nnunet/Dataset007_task2_Ts/imagesTs \
-o /mnt/hdd2/task2/nnunet/predict_results/test_func_fast \
-d 007 -c 2d -f 0 \
--use_gaussian
"""

import argparse
import torch
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, isdir, join
from nnunetv2.utilities.file_path_utilities import get_output_folder

# use for calculate inference time
from nnunetv2.inference.predict_from_raw_data_fast_cal_time import FastNNUNetPredictor
# from nnunetv2.inference.predict_from_raw_data_fast import FastNNUNetPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast nnU-Net inference (single fold, optional gaussian).")
    parser.add_argument('-i', type=str, required=True, help='Input folder (PNG images).')
    parser.add_argument('-o', type=str, required=True, help='Output folder for segmentations.')
    parser.add_argument('-d', type=str, required=True, help='Dataset id or name (used to resolve model folder).')
    parser.add_argument('-c', type=str, required=True, help='Configuration name (e.g., 2d, 3d_lowres, 3d_fullres).')
    parser.add_argument('-f', type=int, default=0, help='Fold to use (default 0).')
    parser.add_argument('-chk', type=str, default='checkpoint_final.pth', help='Checkpoint filename inside fold_X folder.')
    parser.add_argument('--use_gaussian', action='store_true', help='Enable Gaussian blending for sliding-window patches.')
    parser.add_argument('-device', type=str, default='cuda', help="Device to use: 'cuda', 'cpu', or 'mps'.")
    parser.add_argument('--pattern', type=str, default='*.png', help="Glob pattern for input files (default '*.png').")
    parser.add_argument('--write_dtype', type=str, default=None, help="Optional dtype for output (e.g. 'uint8').")
    args = parser.parse_args()

    # resolve device
    device = torch.device(args.device)

    # resolve model folder (follows nnUNet folder naming helpers)
    model_folder = get_output_folder(args.d, 'nnUNetTrainer', 'nnUNetPlans', args.c)

    # ensure output folder exists
    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    
    
    # instantiate fast predictor
    predictor = FastNNUNetPredictor(device=device,
                                    use_gaussian=args.use_gaussian,
                                    use_mirroring=False,
                                    tile_step_size=0.5,
                                    verbose=False,
                                    verbose_preprocessing=False,
                                    allow_tqdm=False)

    # initialize model (single fold)
    predictor.initialize_from_trained_model_folder(
        model_folder, 
        (args.f,), 
        args.chk
      )

    # optional dtype parsing
    write_dtype = None
    if args.write_dtype is not None:
        try:
            import numpy as _np
            write_dtype = getattr(_np, args.write_dtype)
        except Exception:
            print(f"Warning: couldn't parse write dtype '{args.write_dtype}'. Falling back to uint8.")
            import numpy as _np
            write_dtype = _np.uint8

    # run predictions over the folder
    predictor.predict_from_folder(args.i, args.o, file_pattern=args.pattern,
                                  num_processes_preprocessing=1,
                                  write_dtype=write_dtype)
