
import pandas as pd
import numpy as np
import os
import joblib # For persistent storage of the model object

# Experimentation and tracking tools
import mlflow as tracker
import mlflow.sklearn
# Core ML libraries
from sklearn.tree import DecisionTreeClassifier as TreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
# *** CHANGED TO RANDOMIZED SEARCH ***
from sklearn.model_selection import RandomizedSearchCV as HyperparameterOptimizer
# *** ADDED scipy.stats FOR RANDOMIZED SEARCH DISTRIBUTIONS ***
from scipy.stats import randint as sp_randint
from sklearn.pipeline import Pipeline as EstimatorPipeline
from sklearn.preprocessing import StandardScaler as FeatureScaler
# For publishing the final model
from huggingface_hub import HfApi

# --- Configuration & Constants ---
# Hugging Face API setup
huggingface_api = HfApi(token=os.getenv("HF_PM_TOKEN"))

# Define model storage paths
OUTPUT_DIR = "."

DATA_SOURCE_REPO = "ShaksML/predictive_maintenance_data"
MODEL_TARGET_REPO = "ShaksML/dt-model-predective-maintenance"

X_TRAIN_FILE = os.path.join(OUTPUT_DIR, "training_features.csv")
X_TEST_FILE = os.path.join(OUTPUT_DIR, "testing_features.csv")
Y_TRAIN_FILE = os.path.join(OUTPUT_DIR, "training_targets.csv")
Y_TEST_FILE = os.path.join(OUTPUT_DIR, "testing_targets.csv")

PERSISTENT_MODEL_FILE = '/content/optimized_dt_predictor.pkl'

# MLflow setup (Assuming the tracking server is already running and configured)
tracker.set_experiment("Predictive_Engine_Analysis")

# Step 1: Ingest Data Splits

try:
    # Load the features and targets
    X_train_data = pd.read_csv(X_TRAIN_FILE)
    X_test_data = pd.read_csv(X_TEST_FILE)
    y_train_series = pd.read_csv(Y_TRAIN_FILE).iloc[:, 0]
    y_test_series = pd.read_csv(Y_TEST_FILE).iloc[:, 0]

    print("All required data segments successfully loaded.")

except Exception as data_load_error:
    print(f"CRITICAL ERROR: Data files could not be read. Check the local paths. Detail: {data_load_error}")
    raise SystemExit("Exiting script because the data simply refused to cooperate.")


# Step 2: Define Model Pipeline and Hyperparameter Space

modeling_pipeline = EstimatorPipeline([
    ('feature_normalization', FeatureScaler()),
    ('classifier', TreeClassifier(random_state=42))
])

# Hyperparameter change: Defined a search space relevant for Decision Trees
# Since we are using RandomizedSearchCV, we define some ranges using sp_randint
hyperparam_space = {
    'classifier__max_depth': [5, 10, 15, None], # Discrete values still work
    # Use sp_randint for integer ranges in RandomizedSearchCV
    'classifier__min_samples_split': sp_randint(2, 20),
    'classifier__criterion': ['gini', 'entropy']
}


# Step 3: Optimize and Track the Model
print("\n--- Initiating Hyperparameter Optimization and MLflow Tracking ---")

# Start an MLflow run for the entire tuning process
with tracker.start_run(run_name="DecisionTree_Predictor") as primary_run: # Updated run name

    # Define the search strategy (RandomizedSearch with F1 scoring)
    optimizer = HyperparameterOptimizer(
        estimator=modeling_pipeline,
        param_distributions=hyperparam_space, # Note: using param_distributions for RandomizedSearchCV
        n_iter=10, # *** RANDOMLY SAMPLE ONLY 10 COMBINATIONS ***
        scoring='f1',
        cv=3,
        verbose=2, # Increased verbosity to see progress
        random_state=42,
        n_jobs=-1
    )

    # Execute the training and tuning
    optimizer.fit(X_train_data, y_train_series)

    optimal_estimator = optimizer.best_estimator_
    best_configuration = optimizer.best_params_

    # Log the key optimization details
    tracker.log_params(best_configuration)
    tracker.log_param("optimization_metric", "f1")
    tracker.log_param("model_type", "DecisionTree") # Updated model type logging


# Step 4: Validate Performance and Log Results
predictions = optimal_estimator.predict(X_test_data)
# Need probabilities for the AUC metric
prob_scores = optimal_estimator.predict_proba(X_test_data)[:, 1]

# Calculate and store all relevant metrics
evaluation_metrics = {
    "acc": accuracy_score(y_test_series, predictions),
    "prec": precision_score(y_test_series, predictions),
    "rec": recall_score(y_test_series, predictions),
    "f1": f1_score(y_test_series, predictions),
    "roc_auc": roc_auc_score(y_test_series, prob_scores)
}

print("\n--- Optimal Model Performance on Validation Set ---")
for metric_name, value in evaluation_metrics.items():
    print(f"{metric_name.upper()}: {value:.4f}")

# Log final metrics and model artifact to the MLflow primary run
with tracker.start_run(run_id=primary_run.info.run_id):
    for metric_name, value in evaluation_metrics.items():
        tracker.log_metric(f"validation_{metric_name}", value)

    # Save the model persistently on the local filesystem
    joblib.dump(optimal_estimator, PERSISTENT_MODEL_FILE)
    print(f"\nModel artifact saved locally: {PERSISTENT_MODEL_FILE}")

    # Log the model to MLflow registry
    mlflow.sklearn.log_model(
        sk_model=optimal_estimator,
        artifact_path="final_dt_asset", # Updated artifact path name
        registered_model_name="EnginePredictorDecisionTree" # Updated registered model name
    )
    print("Model successfully registered with MLflow.")

# Step 5: Publish the Model to Hugging Face

try:
    huggingface_api.create_repo(repo_id=MODEL_TARGET_REPO, repo_type="model", exist_ok=True)

    huggingface_api.upload_file(
        path_or_fileobj=PERSISTENT_MODEL_FILE,
        path_in_repo=os.path.basename(PERSISTENT_MODEL_FILE),
        repo_id=MODEL_TARGET_REPO,
        repo_type="model"
    )
    print(f"Model deployed to Hugging Face Model Hub at: {MODEL_TARGET_REPO}")

except Exception as hf_upload_error:
    print(f"Hugging Face deployment encountered a snag. Detail: {hf_upload_error}")
