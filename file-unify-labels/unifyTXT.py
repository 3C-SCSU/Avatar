import glob  # For matching file patterns ('*.txt')
import os  # For interacting with the operating system (files, dirs, paths)
import shutil  # For moving files
import stat  # For changing file permissions


# Function to determine category from the folder name
def get_category_from_folder(folder_name):
    folder_name = (
        folder_name.lower()
    )  # Converts folder name to lowercase for case-insensitive matching
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
    elif "landing" in folder_name:
        return "landing"
    else:
        # Prints a warning if the folder name doesn't match any known category
        print(f"Category not found: {folder_name}")
        return None  # Returns None to indicate an unmatched folder


# Function to change permissions to ensure files/folders are writable and accessible
def change_permissions(path):
    # Walk through the directory structure recursively
    for root, dirs, files in os.walk(path):
        for d in dirs:
            # Build full directory path
            full_dir = os.path.join(root, d)
            # Get current permission mode
            mode = os.stat(full_dir).st_mode
            # Add user write (W) and execute (X) permissions to directories
            os.chmod(full_dir, mode | stat.S_IWUSR | stat.S_IXUSR)
        for f in files:
            # Build full file path
            full_file = os.path.join(root, f)
            # Get current permission mode
            mode = os.stat(full_file).st_mode
            # Add user write (W) permission to files
            os.chmod(full_file, mode | stat.S_IWUSR)

    top_mode = os.stat(path).st_mode
    flags = stat.S_IWUSR
    if os.path.isdir(path):
        flags |= stat.S_IXUSR
    os.chmod(path, top_mode | flags)


# Function to check if a given path is inside the 'processed/' folder
def is_inside_processed(path, base_dir):
    relative_path = os.path.relpath(path, base_dir)  # Get path relative to base_dir
    return relative_path.startswith(
        "processed" + os.sep
    )  # Check if it starts with "processed/"


# Main function that organizes all .txt files into categorized folders inside "processed"
def move_any_txt_files(base_dir):
    change_permissions(
        base_dir
    )  # Ensure files and folders are writable before processing

    processed_dir = os.path.join(
        base_dir, "processed"
    )  # Define the path to the 'processed' directory
    os.makedirs(
        processed_dir, exist_ok=True
    )  # Create the 'processed' folder if it doesn't exist

    # Define glob pattern to find all .txt files recursively
    pattern = os.path.join(base_dir, "**", "*.txt")

    # Iterate through all matched .txt files
    for txt_path in glob.glob(pattern, recursive=True):
        if is_inside_processed(txt_path, base_dir):
            continue  # Skip if file is already inside 'processed/'

        # Get the name of the folder containing the .txt file
        parent_folder = os.path.basename(os.path.dirname(txt_path))
        category = get_category_from_folder(parent_folder)  # Determine the category

        if not category:
            print(
                f"Skipping (no category match): {txt_path}"
            )  # Skip if no valid category found
            continue

        # Build the target folder path within 'processed/'
        target_dir = os.path.join(processed_dir, category)
        os.makedirs(
            target_dir, exist_ok=True
        )  # Create category folder if not already present

        # Build destination path for the file
        original_name = os.path.basename(txt_path)
        dest = os.path.join(target_dir, original_name)

        # If a file with the same name exists, append _1, _2, etc., to avoid overwriting
        if os.path.exists(dest):
            base, ext = os.path.splitext(original_name)  # Split filename and extension
            i = 1
            new_name = f"{base}_{i}{ext}"  # Create new name with _1
            dest = os.path.join(target_dir, new_name)
            while os.path.exists(
                dest
            ):  # Increment until a non-conflicting name is found
                i += 1
                new_name = f"{base}_{i}{ext}"
                dest = os.path.join(target_dir, new_name)

        # Move the file to its destination inside the processed category
        shutil.move(txt_path, dest)
        print(f"Moved: {txt_path} → {dest}")

    print("TXT unification complete. Cleaning up...")

    # Cleanup phase: remove non-txt files and empty directories (excluding 'processed')
    for root, dirs, files in os.walk(base_dir, topdown=False):
        if os.path.abspath(root).startswith(os.path.abspath(processed_dir)):
            continue  # Skip cleanup inside the 'processed' folder

        for fname in files:
            if not fname.lower().endswith(".txt"):
                path = os.path.join(root, fname)
                os.remove(path)  # Delete non-text files
                print(f"Removed non-text file: {path}")

        if not os.listdir(root):
            os.rmdir(root)  # Remove empty directories
            print(f"Removed empty directory: {root}")

    print("Cleanup complete.")
    print(f"Finished processing directory: {base_dir}")


# Script entry point
if __name__ == "__main__":
    base_directory = "data"  # Set the root folder to be processed
    move_any_txt_files(base_directory)  # Run the main logic
