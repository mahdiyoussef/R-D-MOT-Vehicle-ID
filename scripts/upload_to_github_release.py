import os
import glob
import requests
import getpass
from pathlib import Path

# GitHub Repository details
OWNER = "mahdiyoussef"
REPO = "R-D-MOT-Vehicle-ID"
TAG_NAME = "v5.0"
RELEASE_NAME = "v5.0 SOTA Pipeline Weights"
RELEASE_BODY = "This release contains all the necessary pre-trained weights (YOLO, DINOv2, OSNet, ControlNet) for the v5.0 Pipeline."

def upload_models_to_release():
    print(f"Uploading models to {OWNER}/{REPO} - Release: {TAG_NAME}")
    
    # 1. Get Token
    token = getpass.getpass(prompt="Enter your GitHub Personal Access Token (PAT): ")
    if not token.strip():
        print("Token is required!")
        return

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    # 2. Check if release exists, if not, create it
    api_base_url = f"https://api.github.com/repos/{OWNER}/{REPO}"
    
    # Check token validity first
    user_res = requests.get("https://api.github.com/user", headers=headers)
    if user_res.status_code != 200:
        print(f"ERROR 401: Invalid Token! Please make sure you copied the token correctly.")
        return
    
    # Try to get release by tag
    response = requests.get(f"{api_base_url}/releases/tags/{TAG_NAME}", headers=headers)
    
    if response.status_code == 200:
        release_data = response.json()
        release_id = release_data["id"]
        upload_url = release_data["upload_url"].split("{")[0]
        print(f"Found existing release {TAG_NAME} (ID: {release_id})")
    else:
        # Create release
        print(f"Creating new release {TAG_NAME}...")
        payload = {
            "tag_name": TAG_NAME,
            "name": RELEASE_NAME,
            "body": RELEASE_BODY,
            "draft": False,
            "prerelease": False
        }
        res = requests.post(f"{api_base_url}/releases", headers=headers, json=payload)
        if res.status_code != 201:
            if res.status_code == 404:
                print("\n❌ ERREUR 404: Permission refusée (Not Found).")
                print("Cela signifie que votre Token est valide mais n'a PAS l'autorisation d'écrire dans ce dépôt.")
                print("Solution : Retournez sur GitHub et assurez-vous de bien cocher la case principale 'repo' (Full control of private repositories) lors de la création du Token !")
            else:
                print(f"Failed to create release: {res.json()}")
            return
        
        release_data = res.json()
        release_id = release_data["id"]
        upload_url = release_data["upload_url"].split("{")[0]
        print(f"Created release {TAG_NAME} (ID: {release_id})")

    # 3. Find all model files
    model_dir = Path("models")
    if not model_dir.exists():
        print("Models directory not found!")
        return
        
    # Get all .pt, .pth, .onnx files
    model_files = []
    for ext in ("*.pt", "*.pth", "*.onnx"):
        model_files.extend(model_dir.rglob(ext))

    if not model_files:
        print("No model weights found to upload.")
        return

    # 4. Upload each file
    print(f"Found {len(model_files)} model files. Starting upload...")
    for file_path in model_files:
        file_name = file_path.name
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"Uploading {file_name} ({file_size_mb:.2f} MB)...", end=" ", flush=True)
        
        with open(file_path, "rb") as f:
            headers_upload = headers.copy()
            headers_upload["Content-Type"] = "application/octet-stream"
            
            upload_res = requests.post(
                f"{upload_url}?name={file_name}",
                headers=headers_upload,
                data=f
            )
            
            if upload_res.status_code == 201:
                print("OK!")
            else:
                print(f"FAILED! {upload_res.status_code}")
                try:
                    print(upload_res.json())
                except:
                    pass

    print("\nAll uploads completed! View your release here:")
    print(f"https://github.com/{OWNER}/{REPO}/releases/tag/{TAG_NAME}")

if __name__ == "__main__":
    upload_models_to_release()
