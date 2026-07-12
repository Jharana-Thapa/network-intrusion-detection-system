import os
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")

def download_file(url, dest_path):
    """Downloads a file from a URL and saves it to dest_path."""
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def ensure_datasets():
    """Ensures training and testing dataset files are present locally."""
    success = True
    if not os.path.exists(TRAIN_PATH):
        success = success and download_file(TRAIN_URL, TRAIN_PATH)
    else:
        print(f"Training dataset already exists at {TRAIN_PATH}")

    if not os.path.exists(TEST_PATH):
        success = success and download_file(TEST_URL, TEST_PATH)
    else:
        print(f"Testing dataset already exists at {TEST_PATH}")
        
    return success

if __name__ == "__main__":
    ensure_datasets()