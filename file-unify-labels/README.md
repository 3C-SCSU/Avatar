### Overview

This Python program, `unifyTXT.py` is designed to organize CSV files from various subdirectories into a central directory structure based on specific categories, such as `backward`, `forward`, `landing`, `left`, `right`, and `takeoff`. The program processes directories containing BCI data, identifies files, moves them to appropriate folders, and deletes unnecessary `.csv` files.

### Features

- Organizes TXT Files: Files are moved to corresponding directories like takeoff, backward, right, etc.
- Removes Duplicate Files: Duplicate `.txt` files are identified and deleted.
- Deletes CSV Files: All `.csv` files in the source directories are deleted.
- Cleans Up Empty Directories: Once the files are moved, empty subdirectories and group directories are removed.

### Usage

`unifyTXT.py` should be placed in the same directory that contains the `data/` folder.

The script is configured to look for `data/` by default:
```python
base_directory = "data"
```

To run:
```bash
python unifyTXT.py
```

After running, `.txt` files will be grouped under the `data/` folder by category. Example:
```
data/
├── backward/
├── takeoff/
├── forward/
```
`unifyTXT.py` should be ran inside the same directory that the data directory is located. Line 81 of the program, `base_directory = "data"`, holds the name of the data folder. "data" can be changed depending on the name of the data folder.


---

### Example Directory Structure

Before:
```
data/
  group03/
    individual10/
      takeoff/
        BrainFlow-RAW_2025-02-28_11-59-05_5.txt
        BrainFlow-RAW_2025-02-28_11-59-05_5.csv
  group04/
    Test 5/
      OpenBCISession_backward_5/
        BrainFlow-RAW_backward_5_10.txt
        BrainFlow-RAW_backward_5_10.csv
```

After:
```
data/
  takeoff/
    BrainFlow-RAW_2025-02-28_11-59-05_5.txt
  backward/
    BrainFlow-RAW_backward_5_10.txt
```

---

### Category Recognition

The script recognizes the following folder name patterns (case-insensitive):

- `takeoff`, `take_off`
- `backward`, `backwards`
- `forward`
- `landing`, `land`
- `left`
- `right`

The keyword can appear anywhere in the folder name — for example:
- `OpenBCISession_backward_5/`
- `TAKE_OFF_session01/`

---


### Author

Modified by Group 10 — Brittney Johnson, Veejay Deonarine, and Tamunotekena Ogan  
Fixes Ticket #197 
Original Author: Thomas Herold


