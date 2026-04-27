import pandas as pd
import os

# Define the file paths
base_path = "/mnt/hdd2/task2/nnunet/predict_results/"
patients = ["19", "24", "71", "76", "78"]
input_files = [f"comparison_{p}_metrics.csv" for p in patients]

# Define class groups
instrument_classes = list(range(1, 26))
organ_classes = [26, 27, 28]

def calculate_metrics(df, classes):
    # Filter the dataframe for the specified classes
    group_df = df[df['class'].isin(classes)]
    
    if group_df.empty:
        return {
            "Mean IOU": 0.0,
            "Mean Dice": 0.0,
            "Mean HD95": 0.0
        }
    
    # Calculate mean metrics
    # Note: HD95 might have very large values or NaNs depending on the evaluation script
    # We take the mean across all rows in the group
    metrics = {
        "Mean IOU": group_df['IOU'].mean(),
        "Mean Dice": group_df['Dice'].mean(),
        "Mean HD95": group_df['HD95'].mean()
    }
    return metrics

results_summary = []

for p in patients:
    input_path = os.path.join(base_path, f"comparison_{p}_metrics.csv")
    output_path = os.path.join(base_path, f"comparison_{p}_organ_instrument.csv")
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        continue
        
    df = pd.read_csv(input_path)
    
    organ_metrics = calculate_metrics(df, organ_classes)
    instrument_metrics = calculate_metrics(df, instrument_classes)
    
    # Create the result dataframe
    result_data = {
        "Group": ["Organ", "Instrument"],
        "Mean IOU": [organ_metrics["Mean IOU"], instrument_metrics["Mean IOU"]],
        "Mean Dice": [organ_metrics["Mean Dice"], instrument_metrics["Mean Dice"]],
        "Mean HD95": [organ_metrics["Mean HD95"], instrument_metrics["Mean HD95"]]
    }
    
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    # Store for global summary output
    results_summary.append({
        "Patient": p,
        "Organ_IOU": organ_metrics["Mean IOU"],
        "Organ_Dice": organ_metrics["Mean Dice"],
        "Organ_HD95": organ_metrics["Mean HD95"],
        "Inst_IOU": instrument_metrics["Mean IOU"],
        "Inst_Dice": instrument_metrics["Mean Dice"],
        "Inst_HD95": instrument_metrics["Mean HD95"]
    })

# Print a nice summary table
summary_df = pd.DataFrame(results_summary)
print("\nFinal Summary for Fold 0:")
print(summary_df.to_string(index=False))
