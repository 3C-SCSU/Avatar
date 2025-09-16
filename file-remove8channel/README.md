# Remove 8-Channel TXT Files

This Python script recursively scans a specified directory for `.txt` files that contain the line:
```bash
%Number of channels = 8
```

These files are deleted, and a detailed report is generated listing all deleted files and the total count. The script also ensures appropriate permissions are set to allow file access and deletion.

## Features

- Recursively scans directories and subdirectories
- Detects `.txt` files containing 8-channel data
- Deletes matching files safely
- Generates a timestamped report of all deleted files
- Automatically adjusts file and directory permissions to allow deletion

## Usage

### 1. Place Your Data

Ensure your folder (e.g., `data/`) contains the `.txt` files you want to scan.

### 2. Run the Script

```bash
python remove_8channel_txt.py
base_directory = "data"  # Change this to your target directory
```

By default, the script looks for a folder named data in the current working directory. To change the directory, modify the base_directory

### 3. Output

All .txt files containing 8-channel data will be deleted.

A report named remove-8channel-Report.txt (or remove-8channel-Report_01.txt, etc.) will be created in the same directory.

The report includes:

Timestamp

Number of deleted .txt files

Paths of all deleted files

```bash
project/
│
├── data/  # Folder containing your text files
│   ├── file1.txt
│   ├── subdir/
│       ├── file2.txt
│
├── remove_8channel_txt.py    # The main script
├── remove-8channel-Report.txt # Auto-generated report (created after running)
```

Notes
- The scipt only deletes .txt files that contain the exact line:
```bash
%Number of channels = 8
```
- It skips files it cannot read (due to encoding or permission errors) and logs those errors to the console.

- If a report already exists, the script automatically appends a number to avoid overwriting.
