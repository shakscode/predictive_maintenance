import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# --- Configuration ---
# API token is retrieved from the environment for security
hf_api_handler = HfApi(token=os.getenv("HF_PM_TOKEN"))


HF_DATA_REPO = "ShaksML/predictive_maintenance_data"
DATASET_PATH = f"hf://datasets/{HF_DATA_REPO}/engine_data.csv"


# Define the mapping to standardize column names
COLUMN_RENAME_MAP = {
    'Engine rpm': 'Engine_RPM',
    'Lub oil pressure': 'Lub_Oil_Pressure',
    'Fuel pressure': 'Fuel_Pressure',
    'Coolant pressure': 'Coolant_Pressure',
    'lub oil temp': 'Lub_Oil_Temperature',
    'Coolant temp': 'Coolant_Temperature',
    'Engine Condition': 'Target_Condition'
}


# =============================
# 1. Acquire and Cleanse Data
# =============================
try:

    data_frame = pd.read_csv(DATASET_PATH)

    # Standardize column names
    data_frame.rename(columns=COLUMN_RENAME_MAP, inplace=True)
    print("Source data frame acquired and columns standardized.")


    data_frame['Target_Condition'] = data_frame['Target_Condition'].astype(int)
    print("Target variable data type confirmed.")

except Exception as e:
    print(f"Failed to load dataset from {DATASET_PATH}. Error: {e}")




# 2. Separation and Train/Test Splitting


target_feature = "Target_Condition"
all_features = data_frame.drop(columns=[target_feature])
target_vector = data_frame[target_feature]

# Divide the dataset into training and validation sets
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(
    all_features, target_vector, test_size=0.20, random_state=42, stratify=target_vector
)

print("Data successfully partitioned into Train and Test sets.")


# 3. Persist and Publish Artifacts


# Define the list of splits and their corresponding filenames
split_artifacts = [
    (X_train_set, "training_features.csv"),
    (X_test_set, "testing_features.csv"),
    (y_train_set, "training_targets.csv"),
    (y_test_set, "testing_targets.csv"),
]

for dataset, file_name in split_artifacts:

    # Save the split to the local artifacts folder
    dataset.to_csv(file_name, index=False)

    # Upload the split to the Hugging Face dataset repository
    try:
        hf_api_handler.upload_file(
            path_or_fileobj=file_name,
            path_in_repo=file_name,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
        )
        print(f"Successfully saved locally and uploaded: {file_name}")
    except Exception as upload_error:
        print(f"Could not publish {file_name} to Hugging Face. Error: {upload_error}")
