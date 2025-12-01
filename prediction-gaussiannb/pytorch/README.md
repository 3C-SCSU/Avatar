## Brainwaves Dataset Augmentation Tool
This Python script automates data augmentation for a Brain-Computer Interface (BCI) dataset located in the `brainwaves/` directory. It increases the dataset size by 60% through controlled file replication across 6 specific thought categories: forward, backward, land, takeoff, right, and left. The tool ensures balanced augmentation, anonymization via shuffling, and detailed reporting.

## Features

- **Data Sanitization**: Renames all CSV files to sequential numbers (1.csv, 2.csv, etc.) across categories for anonymization.
- **Balanced Replication**: Randomly replicates files evenly, 50 at a time per category, tracking usage to avoid duplicates until originals are exhausted.
- **Size Monitoring**: Tracks copied files until a 60% increase in total dataset size is achieved.
- **File Organization**: Uses temporary directories for staging, then integrates copies into original category folders.
- **Final Shuffling**: Renames all files to `renamed_1.csv`, `renamed_2.csv`, etc., and shuffles order for additional anonymization.
- **Reporting**: Outputs final dataset size (in GB), replicated files per category, and total file count.

## Usage

1. Place the script (`opendata.py`) in `Avatar/file-opendata/`.
2. Ensure the `brainwaves/` directory contains subdirectories like `brainwave_rawdata_spring2024/` with category folders (forward, backward, etc.) holding `.csv` files.
3. Run: `python opendata.py`
4. Review console output for completion stats.

The script processes ~4000 original files (~2.2GB compressed) to reach ~3GB total.

## Directory Structure

```
Avatar/file-opendata/
├── opendata.py          # This script
└── brainwaves/          # Input/output directory
    └── brainwave_rawdata_*  # Year/semester dirs
        ├── forward/
        ├── backward/
        ├── land/
        ├── takeoff/
        ├── right/
        └── left/
```

Temporary `tmp_raw_data/` is created and cleaned up automatically.

## Key Logic

- Scans and renames originals to numeric format.
- Replicates until 60% size increase: copies unique files randomly, resetting trackers when needed.
- Moves copies to originals, shuffles/renames everything.
- Handles multiple years of data evenly.

## Important Notes

- **Privacy**: Brainwave data requires authorization before sharing. Do not upload raw files to GitHub.
- **Cleanup for Repo**: After processing, remove `brainwaves/` contents and add `access_data.txt`:
  ```
  GITHUB DOES NOT ALLOW UPLOADING LARGE FILES. 
  TO ACCESS OUR OPEN DATABUCKET THROUGH DELTA LAKE, PLEASE CONTACT US. 
  SOURCE: GitHub Free and Pro: Up to 2 GB
  ```
- **Donation**: This code is contributed to the Avatar Open Source project via pull request.
- **Dependencies**: Standard library only (`os`, `pathlib`, `random`, `shutil`, `re`). No external installs needed.

## Output Example

```
----- Data Generation Complete! -----
Total size of the new dataset: 3.12 GB
Number of replicated files per category:
  - ('brainwave_rawdata_spring2024', 'forward'): 50
  ...
Total number of files in the new dataset: 6400
```
