# Author: Modified by Group 10 (Brittney Johnson, Veejay Deonarine, and Tamunotekena Ogan)
# Fixes Ticket #197
# File: unifyTXT.py

import os  # Provides functions for file system interaction
import shutil  # Provides functions for copying, moving, and removing directories/files
import glob  # Provides functions for finding files
import stat  # Provides constants and functions for interpreting file mode bits

# Dynamically map folder names to categories (case insensitive)
def get_category_from_folder(folder_name):
    folder_name = folder_name.lower()
    if "takeoff" in folder_name or "take_off" in folder_name:
        return "takeoff"
    elif "backward" in folder_name or "backwards" in folder_name:
        return "backward"
    elif "right" in folder_name:
        return "right"
    elif "left" in folder_name:
        return "left"
    elif "forward" in folder_name:
        return "forward"
    elif "landing" in folder_name or "land" in folder_name:
        return "landing"
    else:
        print(f"Category not found: {folder_name}")
        return None

# Helper function to fix permission issues (especially for folders like group04)
def change_permissions(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.chmod(os.path.join(root, dir), stat.S_IWRITE)
        for file in files:
            os.chmod(os.path.join(root, file), stat.S_IWRITE)
    os.chmod(path, stat.S_IWRITE)

# Function to remove any remaining empty folders (even after initial pass)
def remove_empty_folders(path):
    for dirpath, dirnames, _ in os.walk(path, topdown=False):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            try:
                if not os.listdir(full_path):
                    change_permissions(full_path)
                    os.rmdir(full_path)
                    print(f"Removed leftover empty folder: {full_path}")
            except Exception as e:
                print(f"Could not remove folder: {full_path} → {e}")

# Walk all subdirectories and process only folders that match categories
def move_files_to_categories(base_dir):
    for group in os.listdir(base_dir):
        group_path = os.path.join(base_dir, group)
        if not os.path.isdir(group_path):
            continue

        for dirpath, _, _ in os.walk(group_path):
            if not os.path.isdir(dirpath):
                continue

            folder_name = os.path.basename(dirpath)
            category = get_category_from_folder(folder_name)
            if category:
                print(f"Processing category: {category}")
                category_path = os.path.join(base_dir, category)
                os.makedirs(category_path, exist_ok=True)

                # Delete all .csv files
                csv_files = glob.glob(os.path.join(dirpath, "*.csv"))
                for csv_file in csv_files:
                    print(f"Deleting .csv file: {csv_file}")
                    os.remove(csv_file)

                # Move all .txt files to the category folder
                txt_files = glob.glob(os.path.join(dirpath, "*.txt"))
                for txt_file in txt_files:
                    filename = os.path.basename(txt_file)
                    destination = os.path.join(category_path, filename)
                    if not os.path.exists(destination):
                        print(f"Moving: {txt_file} to {destination}")
                        shutil.move(txt_file, destination)
                    else:
                        print(f"Removing duplicate .txt file: {txt_file}")
                        os.remove(txt_file)

                # Try to delete the folder if it's empty (immediate cleanup)
                if not os.listdir(dirpath):
                    try:
                        change_permissions(dirpath)
                        os.rmdir(dirpath)
                        print(f"Removed empty folder: {dirpath}")
                    except Exception as e:
                        print(f"Could not remove folder: {dirpath} → {e}")

    print(f"\nAll folders under '{base_dir}' processed. .txt files unified, .csv files deleted, and empty folders removed.")

    # Final pass to remove any leftover empty folders (recursively)
    remove_empty_folders(base_dir)

if __name__ == "__main__":
    base_directory = "data"  # Base directory to start from: \data\. This should be in the same directory as the Python script.
    move_files_to_categories(base_directory)
