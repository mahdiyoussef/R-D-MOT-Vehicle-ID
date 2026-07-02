import os
from huggingface_hub import HfApi, login

def upload_to_huggingface(repo_id: str, folder_path: str = "models/"):
    """
    Uploads the local models directory to a Hugging Face model repository.
    """
    print("Authenticate with your Hugging Face account.")
    login()  # This will prompt for a token
    
    api = HfApi()
    
    # Create the repository if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"Repository {repo_id} is ready.")
    except Exception as e:
        print(f"Could not create repo (maybe it exists): {e}")

    print(f"Uploading files from {folder_path} to {repo_id}...")
    
    # Upload the entire models directory
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="models"  # This will place everything in a 'models' folder in the repo
    )
    print("Upload complete!")

if __name__ == "__main__":
    # Change 'mahdiyoussef/vehicles-tracking-models' to your preferred HF repo name
    repo_name = "mahdiyoussef/vehicles-tracking-models"
    upload_to_huggingface(repo_name)
