#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "gdown>=5.1.0",
# ]
# ///

import os
import sys

try:
    import gdown
except ImportError:
    print("Error: 'gdown' library is not installed.")
    print("Please install it using: pip install gdown")
    sys.exit(1)

def download_models():
    """
    Downloads the pre-trained models from Google Drive.
    The Drive folder contains 'film_net' and 'vgg' subdirectories.
    These will be downloaded into the 'pretrained_models' directory.
    """
    # The Google Drive folder ID/URL containing the models
    url = 'https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy'
    output_dir = 'pretrained_models'

    # Check if models are already downloaded
    if os.path.isdir(os.path.join(output_dir, 'film_net')) and \
       os.path.isdir(os.path.join(output_dir, 'vgg')):
        print(f"Pretrained models already exist in '{output_dir}'. Skipping download.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Downloading models to {output_dir}...")
    
    # gdown.download_folder downloads the entire folder contents
    # use_cookies=False is usually fine for public folders, but verify if needed
    gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
    
    print("Download complete.")
    
    # Verify the expected structure for predict.py
    expected_path = os.path.join(output_dir, 'film_net', 'Style', 'saved_model')
    if os.path.exists(expected_path):
        print(f"Verification successful: {expected_path} exists.")
    else:
        print(f"Warning: {expected_path} does not exist. Please check the downloaded content.")

if __name__ == "__main__":
    download_models()

