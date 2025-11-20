import pandas as pd
import numpy as np
import os
import joblib
import mlflow as tracker
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier as TreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV as HyperparameterOptimizer
from sklearn.pipeline import Pipeline as EstimatorPipeline
from sklearn.preprocessing import StandardScaler as FeatureScaler
from huggingface_hub import HfApi

huggingface_api = HfApi(token=os.getenv("HF_PM_TOKEN"))

OUTPUT_DIR = "."
MODEL_TARGET_REPO = "ShaksML/dt-model-predective-maintenance"

X_TRAIN_FILE = os.path.join(OUTPUT_DIR, "training_features.csv")
X_TEST_FILE = os.path.join(OUTPUT_DIR, "testing_features.csv")
Y_TRAIN_FILE = os.path.join(OUTPUT_DIR, "training_targets.csv")
Y_TEST_FILE = os.path.join(OUTPUT_DIR, "testing_targets.csv")

PERSISTENT_MODEL_FILE = 'optimized_dt_predictor.pkl'


tracker.set_experiment("Predictive_Engine_Analysis")


def train_and_register_dt_model():
    """Loads data, trains an optimized Decision Tree model, logs to MLflow, and uploads to HF."""
    print("Starting Decision Tree model training and tracking...")

    # Step 1: Ingest Data Splits
    try:
        X_train_data = pd.read_csv(X_TRAIN_FILE)
        X_test_data = pd.read_csv(X_TEST_FILE)
        # Assuming the target is the first/only column and needs to be a Series
        y_train_series = pd.read_csv(Y_TRAIN_FILE).iloc[:, 0]
        y_test_series = pd.read_csv(Y_TEST_FILE).iloc[:, 0]
        print("All required data segments successfully loaded.")
    except Exception as data_load_error:
        print(f"CRITICAL ERROR: Data files could not be read. Detail: {data_load_error}")
        return # Exit the function if data fails to load

    # Step 2: Define Model Pipeline and Hyperparameter Space
    modeling_pipeline = EstimatorPipeline([
        ('feature_normalization', FeatureScaler()),
        ('classifier', TreeClassifier(random_state=42))
    ])

    hyperparam_space = {
        'classifier__max_depth': [5, 10, 15, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__criterion': ['gini', 'entropy']
    }

    with tracker.start_run(run_name="DecisionTree_Predictor") as primary_run:
        optimizer = HyperparameterOptimizer(
            estimator=modeling_pipeline,
            param_grid=hyperparam_space,
            scoring='f1',
            cv=3,
            n_jobs=-1
        )

        optimizer.fit(X_train_data, y_train_series)

        optimal_estimator = optimizer.best_estimator_
        best_configuration = optimizer.best_params_
        tracker.log_params(best_configuration)

        predictions = optimal_estimator.predict(X_test_data)
        prob_scores = optimal_estimator.predict_proba(X_test_data)[:, 1]

        evaluation_metrics = {
            "acc": accuracy_score(y_test_series, predictions),
            "prec": precision_score(y_test_series, predictions, zero_division=0),
            "rec": recall_score(y_test_series, predictions, zero_division=0),
            "f1": f1_score(y_test_series, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test_series, prob_scores)
        }

        for metric_name, value in evaluation_metrics.items():
            tracker.log_metric(f"validation_{metric_name}", value)
        print(f"Optimal F1 Score: {evaluation_metrics['f1']:.4f}, AUC: {evaluation_metrics['roc_auc']:.4f}")

        joblib.dump(optimal_estimator, PERSISTENT_MODEL_FILE)
        print(f"Model artifact saved locally: {PERSISTENT_MODEL_FILE}")

        # Log the model to MLflow registry
        mlflow.sklearn.log_model(
            sk_model=optimal_estimator,
            artifact_path="final_dt_asset",
            registered_model_name="EnginePredictorDecisionTree"
        )
        print("Model successfully registered with MLflow.")

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
            print(f"WARNING: Hugging Face deployment encountered a snag. Ensure HF_PM_TOKEN is valid. Detail: {hf_upload_error}")


if __name__ == "__main__":
    train_and_register_dt_model()
