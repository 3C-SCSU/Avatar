### Overview

This Python program, `unifyTXT.py`, is an updated version of the original unification tool created to organize `.txt` files from subdirectories and remove `.csv` files. It was developed by Group 10 to fix Ticket #197. The script walks through all folders inside the `data/` directory and unifies `.txt` files into category folders such as `backward`, `takeoff`, `landing`, etc., while removing `.csv` files and empty folders.

This version corrects a previous limitation where the script only worked with specific folder structures. It now supports all group datasets, regardless of subfolder naming or nesting depth.

---

### Features

- Organizes TXT Files: Moves `.txt` files into central category folders like `takeoff`, `backward`, `left`, etc.
- Deletes CSV Files: All `.csv` files found in subdirectories are deleted.
- Automatically Detects Categories: Recognizes keywords in folder names, even if nested or uniquely labeled (e.g., `OpenBCISession_backward_5`).
- Fixes Permissions: Handles read-only files/folders to prevent errors.
- Cleans Up: Deletes empty folders after moving or deleting files.

---

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

