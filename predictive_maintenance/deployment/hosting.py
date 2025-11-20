from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token=os.getenv("HF_PM_TOKEN"))

repo_id = "ShaksML/predictive_maintenance_shakthi"
repo_type = "space"

# Create the Hugging Face Space if it doesn't exist
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except Exception:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=os.getenv("HF_PM_TOKEN"), space_sdk='docker') # Changed space_sdk to 'docker'
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="predictive_maintenance/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",
)
