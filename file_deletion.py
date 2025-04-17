#!/usr/bin/env python3
import os
import math
import random
from pathlib import Path

def delete_90_percent_tif(folder_path):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: {folder} is not a valid directory.")
        return

    # Recursively find all .tif files (case-insensitive)
    tif_files = [f for f in folder.rglob('*') if f.is_file() and f.suffix.lower() == '.jpg']
    total_files = len(tif_files)
    if total_files == 0:
        print("No .tif files found in the directory.")
        return

    # Randomize the order of the found files
    random.shuffle(tif_files)

    # Calculate the number of files to keep (roughly 10%) and delete (roughly 90%)
    keep_count = math.ceil(total_files * 0.10)
    files_to_delete = tif_files[keep_count:]

    print(f"Total .tif files found: {total_files}")
    print(f"Keeping {keep_count} files and deleting {len(files_to_delete)} files (approximately 90% deletion).")

    # Delete the selected files
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

if __name__ == "__main__":
    # Hard-coded folder path; change this path as needed.
    folder_input = "C:/Users/vlad_/Desktop/10perc - Copy/train+val/train/"
    delete_90_percent_tif(folder_input)
