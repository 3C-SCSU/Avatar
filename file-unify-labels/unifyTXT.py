# Author: Modified by Group 10 (Brittney Johnson, Veejay Deonarine, and Tamunotekena Ogan)
# Fixes Ticket #197
# File: unifyTXT.py

import os  # Provides functions for file system interaction
import shutil  # Provides functions for copying, moving, and removing directories/files
import glob  # Provides functions for finding files
import stat  # Provides constants and functions for interpreting file mode bits

# Dynamically map folder names to categories (case insensitive)
def get_category_from_folder(folder_name):
    folder_name = folder_name.lower()  #Change to lowercase to help find
    if "takeoff" in folder_name or "take_off" in folder_name:  # Checks for unique takeoff names also
        return "takeoff"
    elif "backward" in folder_name or "backwards" in folder_name:  # Checks for unique backward names also
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
        print(f"Category not found: {folder_name}")  # Print message to indicate no category was found
        return None  # Return none to indicate no matching category was found

# Helper function to fix permission issues (especially for folders like group04)
def change_permissions(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.chmod(os.path.join(root, dir), stat.S_IWRITE)
        for file in files:
            os.chmod(os.path.join(root, file), stat.S_IWRITE)
    os.chmod(path, stat.S_IWRITE)

# Process the directory with the BCI data
def move_files_to_categories(base_dir):
    for group in os.listdir(base_dir):  # Loop over each group (not static) subdirectory
        group_path = os.path.join(base_dir, group)
        if not os.path.isdir(group_path):  # Skip if it is not a directory
            continue

        for subdir in os.listdir(group_path):  # Loop over each individual/test (not static) subdirectory
            subdir_path = os.path.join(group_path, subdir)
            if os.path.isdir(subdir_path):

                for folder in os.listdir(subdir_path):  # Loop over each subdirectory (landing, forward, etc.)
                    folder_path = os.path.join(subdir_path, folder)
                    if os.path.isdir(folder_path):
                        category = get_category_from_folder(folder)  # Determine the folder category for unifying (landing,forward,left,etc.)
                        if category:  # Only continue if a valid category is identified
                            print(f"Processing category: {category}")
                            category_path = os.path.join(base_dir, category)
                            os.makedirs(category_path, exist_ok=True)  # Create category folder if it doesn't exist

                            # Delete all .csv files
                            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
                            for csv_file in csv_files:
                                print(f"Deleting .csv file: {csv_file}")
                                os.remove(csv_file)

                            # Move all .txt files to the category folder
                            txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
                            for txt_file in txt_files:
                                filename = os.path.basename(txt_file)
                                destination = os.path.join(category_path, filename)
                                if not os.path.exists(destination):  # Avoid duplicates
                                    print(f"Moving: {txt_file} to {destination}")
                                    shutil.move(txt_file, destination)
                                else:
                                    print(f"Removing duplicate .txt file: {txt_file}")
                                    os.remove(txt_file)

                            # Try to delete folder if empty
                            if not os.listdir(folder_path):
                                try:
                                    change_permissions(folder_path)
                                    os.rmdir(folder_path)
                                    print(f"Removed empty folder: {folder_path}")
                                except Exception as e:
                                    print(f"Could not remove folder: {folder_path} → {e}")

                # Try to delete subdir if empty
                if not os.listdir(subdir_path):
                    try:
                        change_permissions(subdir_path)
                        os.rmdir(subdir_path)
                        print(f"Removed empty subdirectory: {subdir_path}")
                    except Exception as e:
                        print(f"Could not remove folder: {subdir_path} → {e}")

        # Try to delete group folder if empty
        if not os.listdir(group_path):
            try:
                change_permissions(group_path)
                os.rmdir(group_path)
                print(f"Removed empty group directory: {group_path}")
            except Exception as e:
                print(f"Could not remove folder: {group_path} → {e}")

    print(f"\nAll folders under '{base_dir}' processed. .txt files unified, .csv files deleted, and empty folders removed.") #Program completion message

if __name__ == "__main__":
    base_directory = "data"  # Base directory to start from: \data\. This should be in the same directory as the Python script.
    move_files_to_categories(base_directory) #Function call to do all of the data processing

