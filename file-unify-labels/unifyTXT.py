# Author: Thomas Herold
# File: unifyCSV.py

import os  # Provides functions for file system interaction
import shutil  # Provides functions for copying, moving, and removing directories/files
import glob  # Provides functions for finding files
import stat


# Dynamically map folder names to categories (case insensitive)
def get_category_from_folder(folder_name):
    folder_name = folder_name.lower()  # Change to lowercase to help find
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
    elif "landing" in folder_name:
        return "landing"
    else:
        print(f"Category not found: {folder_name}")  # Print message to indicate no category was found
        return None  # Return none to indicate no matching category was found

def change_permissions(path):
    """
    Grant owner-write on every file, and owner-write + owner-execute
    on every directory, under `path`.  Preserves all other bits.
    """
    for root, dirs, files in os.walk(path):
        # Grant write+execute on each directory so we can cd into it
        for d in dirs:
            full_dir = os.path.join(root, d)
            mode = os.stat(full_dir).st_mode
            os.chmod(full_dir, mode | stat.S_IWUSR | stat.S_IXUSR)

        # Grant write on each file so we can move or delete it
        for f in files:
            full_file = os.path.join(root, f)
            mode = os.stat(full_file).st_mode
            os.chmod(full_file, mode | stat.S_IWUSR)

    # Finally, also fix the top‐level path itself
    top_mode = os.stat(path).st_mode
    # if it’s a directory, ensure it’s traversable too:
    flags = stat.S_IWUSR
    if os.path.isdir(path):
        flags |= stat.S_IXUSR
    os.chmod(path, top_mode | flags)

# Process the directory with the BCI data
def move_any_csvs(base_dir):
    #1) Change permissions
    change_permissions(base_dir)

    # 2) Move all .txt files into category folders
    base_directory = base_dir

    pattern = os.path.join(base_dir, '**', '*.txt')
    for txt_path in glob.glob(pattern, recursive=True):
        parent = os.path.basename(os.path.dirname(txt_path))
        category = get_category_from_folder(parent)
        if not category:
            print(f"  Skipping (no category match): {txt_path}")
            continue

        target_dir = os.path.join(base_dir, category)
        os.makedirs(target_dir, exist_ok=True)

        dest = os.path.join(target_dir, os.path.basename(txt_path))
        shutil.move(txt_path, dest)
        print(f"  Moved: {txt_path} → {dest}")


    print("TXT unification complete. Now cleaning up other files & empty dirs…")

    # 3) Delete any non-.txt files
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for fname in files:
            if not fname.lower().endswith('.txt'):
                path = os.path.join(root, fname)
                os.remove(path)
                print(f"  Removed non-txt file: {path}")

        # 4) Remove empty directories
        #    (os.listdir returns [] only if directory is empty)
        if not os.listdir(root):
            os.rmdir(root)
            print(f"  Removed empty directory: {root}")

    print("Cleanup complete.")


    print(f"{base_directory} directory processed, TXT files unified!")  # Program completion message


if __name__ == "__main__":
    base_directory = "data"  # Base directory to start from: \data\. This should be in the same directory as the Python script.
    move_any_csvs(base_directory)  # Function call to do all of the data processing
