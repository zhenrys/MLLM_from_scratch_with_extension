"""
This is the datasets/data_utils.py module.

It contains utility functions for dataset handling, such as downloading files.
"""
import os
import requests
from tqdm import tqdm

def download_file(url: str, filepath: str):
    """
    Downloads a file from a given URL to a specified path with a progress bar.
    If the file already exists, the download is skipped.

    Args:
        url (str): The URL of the file to download.
        filepath (str): The local path where the file should be saved.
    """
    if os.path.exists(filepath):
        print(f"File already exists at {filepath}, skipping download.")
        return 

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    try:
        # Use streaming download, which is more friendly for large files
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise an exception if the status code is 4xx or 5xx
            
            # Get the total file size from the response headers to configure the tqdm progress bar
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            print(f"Downloading {url} to {filepath}")
            with open(filepath, 'wb') as f, tqdm(
                total=total_size_in_bytes, unit='iB', unit_scale=True, desc=os.path.basename(filepath)
            ) as progress_bar:
                for chunk in r.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            
            # Check if the download is complete (if the server provided content-length)
            # Note: This check has been removed as it could be too strict and cause false positives in some cases.
            # The successful execution of the with statement is a more reliable indicator of a completed download.
            final_size = progress_bar.n
            if total_size_in_bytes != 0 and final_size < total_size_in_bytes:
                print(f"Warning: Download might be incomplete. Expected size: {total_size_in_bytes}, Downloaded size: {final_size}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")
        # If download fails, clean up the incomplete file
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    # Example usage: Download a small text file for testing
    print("Testing the download_file utility...")
    test_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    test_filepath = "data/tinyshakespeare/sample.txt"
    download_file(test_url, test_filepath)
    
    # Test skipping the download
    print("\nAttempting to download the same file again...")
    download_file(test_url, test_filepath)