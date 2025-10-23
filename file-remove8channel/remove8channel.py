import os  # for file system access
import stat  # for permission control
from datetime import datetime  # to get the timestamp to add in the report


# Function to change permissions for all files and directories recursively within the given path
def change_permissions(path):
    # Walk through all directories and files inside the given path
    for root, dir, files in os.walk(
        path
    ):  # root -> root directory, dir -> sub directories in the root directory, files -> the files stored in the root and sub directories
        for d in dir:
            full_dir = os.path.join(root, d)  # Get the full path to the directory
            mode = os.stat(full_dir).st_mode  # Get current permission mode
            os.chmod(
                full_dir, mode | stat.S_IWUSR | stat.S_IXUSR
            )  # Add write and execute permissions for the user

        for f in files:
            full_file = os.path.join(root, f)  # Get the full path to the file
            mode = os.stat(full_file).st_mode  # Get current permission mode
            os.chmod(
                full_file, mode | stat.S_IWUSR
            )  # Add write permission for the user

    top_mode = os.stat(path).st_mode
    flags = stat.S_IWUSR
    if os.path.isdir(
        path
    ):  # check whether the path is a directory and then adds execute permission to that as well
        flags |= stat.S_IXUSR
    os.chmod(path, top_mode | flags)  # Apply updated permissions


# Function to delete a given .txt file
def delete_txt_file(path):
    os.remove(path)  # Remove the file from the filesystem


# Function to create a report about deleted .txt files
def create_report(txt_files, txt_file_count, save_dir):
    base_report_name = "remove-8channel-Report"  # Base name for the report file
    extension = ".txt"  # File extension for the report
    report = os.path.join(
        save_dir, base_report_name + extension
    )  # Construct initial report path
    i = 1  # Counter for renaming report if one already exists

    # If the report file already exists, keep incrementing the number until a unique name is found
    while os.path.exists(report):
        report = os.path.join(save_dir, f"{base_report_name}_{i:02d}{extension}")
        i += 1

    # Write the report file
    with open(report, "w") as report_file:
        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )  # Get current timestamp
        report_file.write(f"Report created on: {timestamp}\n\n")  # Write creation time
        report_file.write(
            f"Total .txt files with 8 channel data were: {txt_file_count}\n"
        )  # Total count
        report_file.write("Deleted TXT files were:\n")  # Header
        for t_file in txt_files:  # List each deleted file
            report_file.write(f"\t{t_file}\n")

    # Print confirmation message
    print(f"Report saved as: {report}")


# Main function to find and remove specific .txt files and generate a report
def file_remover(path):
    base_directory = path  # Store base directory path

    try:
        change_permissions(base_directory)
    except Exception as e:
        print(f"Error changing permissions for '{base_directory}': {e}")
        return

    txt_file_count = 0  # Counter for matching .txt files that are being deleted
    txt_files = []  # List to store paths of files to be deleted

    # Traverse through directory and its subdirectories
    for root, dirs, files in os.walk(base_directory):
        for fileName in files:
            filePath = os.path.join(root, fileName)  # Full path to current file
            shouldDelete = False  # Flag to indicate deletion
            deletingFilename = ""  # Store the filename to be deleted

            try:
                # Open and read file line-by-line
                with open(filePath, "r") as f:
                    for line in f:
                        # Check for specific line indicating 8 channel data
                        if "%Number of channels = 8" in line:
                            txt_file_count += 1  # Increment counter
                            txt_files.append(filePath)  # Add to delete list
                            shouldDelete = True  # Mark for deletion
                            deletingFilename = filePath  # Store file path
                            break  # No need to read more lines
            except Exception as e:
                print(f"Error reading file '{filePath}': {e}")
                continue

            if shouldDelete:
                try:
                    delete_txt_file(deletingFilename)
                except Exception as e:
                    print(f"Error deleting file '{deletingFilename}': {e}")
                    continue
    try:
        create_report(txt_files, txt_file_count, base_directory)
    except Exception as e:
        print(f"Error creating report: {e}")  # Report error in report creation


# Entry point when script is run directly
if __name__ == "__main__":
    base_directory = "data"  # Default directory to process
    file_remover(base_directory)
