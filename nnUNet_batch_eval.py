import glob
import logging
import os
import csv
import numpy as np
from PIL import Image
from medpy import metric
import pandas as pd
from multiprocessing import Pool

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_image(path):
    img = Image.open(path)
    return np.array(img)

def calculate_metrics(pred_binary, label_binary, class_id):
    if np.sum(label_binary) == 0:
        return None
    
    intersection = np.logical_and(pred_binary, label_binary)
    union = np.logical_or(pred_binary, label_binary)
    
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    
    tp = np.sum(intersection)
    fp = np.sum(pred_binary) - tp
    fn = np.sum(label_binary) - tp
    
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    try:
        # HD95 计算
        hd95_val = metric.binary.hd95(pred_binary, label_binary) if np.sum(pred_binary) > 0 else np.nan
    except Exception:
        hd95_val = np.nan
        
    return {
        'class': class_id,
        'IOU': iou,
        'Dice': dice,
        'HD95': hd95_val,
        'Precision': precision,
        'Recall': recall
    }

def process_one_image(args):
    image_file, fold_dir, labels_dir = args
    png_name = os.path.basename(image_file).split('_0000.')[0]
    pred_path = os.path.join(fold_dir, "masks", f"{png_name}.png")
    label_path = os.path.join(labels_dir, f"{png_name}.png")
    
    if not os.path.exists(pred_path) or not os.path.exists(label_path):
        return None
    
    try:
        pred_mask = load_image(pred_path)
        label_mask = load_image(label_path)
        
        # 裁剪区域 (与 nnUNet_cal_metrics.py 一致)
        left, upper, right, lower = 289, 0, 1631, 1004
        pred_mask = pred_mask[upper:lower, left:right]
        label_mask = label_mask[upper:lower, left:right]
        
        # 获取真值中存在的所有类别 (排除背景 0)
        unique_labels = np.unique(label_mask)
        valid_classes = unique_labels[unique_labels > 0]
        
        image_results = []
        for cls in valid_classes:
            m = calculate_metrics((pred_mask == cls).astype(np.uint8), 
                                 (label_mask == cls).astype(np.uint8), 
                                 cls)
            if m:
                image_results.append(m)
        
        if image_results:
            return {'filename': os.path.basename(image_file), 'metrics': image_results}
    except Exception as e:
        logger.error(f"Error processing {png_name}: {e}")
    
    return None

def process_fold(fold_idx):
    base_results_dir = "/mnt/hdd2/task2/nnunet/predict_results"
    images_dir = "/mnt/hdd2/task2/nnunet/Dataset007_task2_Ts/imagesTs"
    labels_dir = "/mnt/hdd2/task2/nnunet/Dataset007_task2_Ts/labelsTs"
    
    fold_dir = os.path.join(base_results_dir, f"fold{fold_idx}")
    if not os.path.exists(fold_dir):
        logger.warning(f"Fold directory {fold_dir} not found. Skipping.")
        return

    patients = ["19", "24", "71", "76", "78"]
    instrument_classes = list(range(1, 26))
    organ_classes = [26, 27, 28]

    for patient in patients:
        logger.info(f"Processing Fold {fold_idx}, Patient {patient}...")
        image_files = sorted(glob.glob(os.path.join(images_dir, f"{patient}*.png")))
        
        if not image_files:
            logger.warning(f"No images found for patient {patient} in {images_dir}")
            continue

        # 使用进程池并行计算
        pool_args = [(f, fold_dir, labels_dir) for f in image_files]
        with Pool(processes=8) as pool:
            all_results = pool.map(process_one_image, pool_args)
        
        # 过滤空结果
        all_results = [r for r in all_results if r is not None]
        
        # 保存详细指标 CSV
        metrics_csv = os.path.join(fold_dir, f"comparison_{patient}_metrics.csv")
        with open(metrics_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'class', 'IOU', 'Dice', 'HD95', 'Precision', 'Recall'])
            writer.writeheader()
            for res in all_results:
                for m in res['metrics']:
                    row = {'filename': res['filename'], **m}
                    writer.writerow(row)
        
        # 计算汇总指标 (Organ vs Instrument)
        df = pd.read_csv(metrics_csv)
        summary_csv = os.path.join(fold_dir, f"comparison_{patient}_organ_instrument.csv")
        
        summary_rows = []
        for name, classes in [("Organ", organ_classes), ("Instrument", instrument_classes)]:
            sub_df = df[df['class'].isin(classes)]
            if not sub_df.empty:
                summary_rows.append({
                    "Group": name,
                    "Mean IOU": sub_df['IOU'].mean(),
                    "Mean Dice": sub_df['Dice'].mean(),
                    "Mean HD95": sub_df['HD95'].mean()
                })
        
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
            logger.info(f"Saved: {summary_csv}")

if __name__ == "__main__":
    # 处理 Fold 1 到 4
    for f_idx in [1, 2, 3, 4]:
        process_fold(f_idx)
    logger.info("All folds processed successfully.")
