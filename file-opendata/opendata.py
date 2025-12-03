import os 
from pathlib import Path
import random
import shutil
import re

# -----------------------------
# Setup paths
# -----------------------------

# Get the directory of the current script
base_dir = Path(__file__).resolve().parent

# List all subdirectories inside "brainwaves"
raw_data_dir = [d for d in (base_dir / "brainwaves").iterdir() if d.is_dir()]


# Temporary directory to store copied files
tmp_raw_data_dir = Path("tmp_raw_data")

# List of 6 categories
category_list = ['forward', 'backward', 'land', 'takeoff', 'right', 'left']

# List of original raw data files
all_file_list = []

# Dictionary to store the max file number for each category
total_original_files = {}

# Dictionary to count of copied files
replicated_counts = {}

# -----------------------------------
# Renaming all files to numeric names 
# -----------------------------------
def rename_all_files_to_numbers():
    for raw_dir in raw_data_dir:
        for category in raw_dir.iterdir():
            # Skip directories that are not in the target category list
            if category.name not in category_list or not category.is_dir():
                continue
            # Get all files in the folder in the current order
            files = [f for f in category.iterdir()
                     if f.is_file() and f.suffix.lower() == ".csv"]
            # Rename files to sequential numbers: 1.csv, 2.csv, 3.csv, ...
            for i, file in enumerate(files, 1):
                new_path = file.with_name(f"{i}.csv")
                os.rename(file, new_path)

# -----------------------------
# Scan raw data directories
# -----------------------------
def scan_raw_data():
  for raw_data in raw_data_dir:
    year = raw_data.name # e.g., "brainwave_rawdata_spring2024"

    for category_dir in raw_data.iterdir():
      # Process only 6 categories and directories
      if category_dir.name not in category_list or not category_dir.is_dir():
                continue
      # Scan all files in the category
      file_numbers = []
      for file_path in category_dir.iterdir():
          if file_path.is_file() and file_path.suffix.lower() == ".csv":
              all_file_list.append(file_path)
              file_numbers.append(int(file_path.stem))
      # Store the max file number for this category
      if file_numbers:
          total_original_files[(year, category_dir.name)] = len(file_numbers)

# -----------------------------
# Define file replication function
# -----------------------------
def replicate_files():
  """
    Replicate files randomly for each category and save them in a temporary directory.
  """
  global total_copied_files_count
  global replicated_counts

  for (year_dir, category_dir), value in total_original_files.items():
      # Define the destination directory
      tmp_dest_dir = tmp_raw_data_dir / year_dir / category_dir

      # Create the directory if it does not exist
      if not tmp_dest_dir.exists():
        tmp_dest_dir.mkdir(parents=True, exist_ok=True)

      # Replicate files 50 times per category
      for _ in range(50):
        # Reset if all files have been used
        if len(replicated_files[(year_dir, category_dir)]) >= value:
          replicated_files[(year_dir, category_dir)].clear()

        # Select an unused random number that has not been used yet
        while True:
          random_number = random.randint(1, value)
          if random_number not in replicated_files[(year_dir, category_dir)]:
              break

        # Set the file number as used
        replicated_files[(year_dir, category_dir)].add(random_number)

        # Define the source file path
        source_file = base_dir / "brainwaves" / year_dir / category_dir / f"{random_number}.csv"

        # Copy the file to the temporary directory
        if source_file.exists():
          # To avoid duplicating the same file, use a unique filename
          total_copied_files_count += 1
          shutil.copy(source_file, tmp_dest_dir / f"{random_number}_copy{total_copied_files_count}.csv")

          # Update the count of copied files
          key = (year_dir, category_dir)
          if key not in replicated_counts:
            replicated_counts[key] = 0
          replicated_counts[key] += 1
        else:
          print(f"Source file not found: {source_file}")

# -----------------------------
# File rename
# -----------------------------
rename_all_files_to_numbers()

# -----------------------------
# Scan data
# -----------------------------
scan_raw_data()

# -----------------------------
# Prepare for file replication
# -----------------------------
total_copied_files_count = 0
# The number of additional files needed to achieve 60% increase)
total_target_files = int(len(all_file_list) * 0.6)  

# Dictionary to store files already replicated in each category
replicated_files = {}
for key in total_original_files.keys():
    replicated_files[key] = set()

# -----------------------------
# Run replication until 60% increase is achieved
# -----------------------------

while total_copied_files_count < total_target_files:
  replicate_files()

# -----------------------------
# Move replicated files from tmp_raw_data to raw_data(the original directories)
# -----------------------------

copied_file_counts = {}

for year_dir in tmp_raw_data_dir.iterdir():
  # Skip if it's not directory
  if not year_dir.is_dir():  
    continue

  year_counts = {}

  for category_dir in year_dir.iterdir(): 
    # Skip if it's not directory
    if not category_dir.is_dir():
      continue
    # Only process 6 categories
    if category_dir.name in category_list and category_dir.is_dir(): 
      # Count files before moving
      file_count = len(list(category_dir.glob("*.csv")))
      year_counts[category_dir.name] = file_count

      # Define the destination directory
      dest_dir = base_dir / "brainwaves" / year_dir.name / category_dir.name
      dest_dir.mkdir(parents=True, exist_ok=True)

      # Move all files from temporary directory to the destination
      for file_path in category_dir.iterdir():
        if file_path.is_file():
            shutil.move(str(file_path), str(dest_dir / file_path.name))

  copied_file_counts[year_dir.name] = year_counts

# -----------------------------
# Shuffle
# -----------------------------

for raw_data in raw_data_dir:
  for category_dir in raw_data.iterdir():
    # Only process 6 target categories
    if category_dir.name not in category_list or not category_dir.is_dir():
      continue

    # Get all files
    all_files = list(category_dir.iterdir())

    # Only process CSV files
    csv_files = []
    for file in all_files:
      if file.is_file() and file.suffix.lower() == ".csv":
        csv_files.append(file)

    # Shuffle the file list
    random.shuffle(csv_files)

    for i, file in enumerate(csv_files, 1):  # Sequential number from 1
      new_name = f"renamed_{i}.csv"
      new_path = file.parent / new_name
      os.rename(file, new_path)

# -----------------------------
# Output
# -----------------------------

print("----- Data Generation Complete! -----")
print("--------------------------------------------------")
# Total size of the new dataset
total_size_bytes = 0
# Total number of new files in the new dataset
total_files = 0

for raw_data in raw_data_dir:
    for category_dir in raw_data.iterdir():
        if category_dir.name in category_list and category_dir.is_dir():
            total_files += len(list(category_dir.glob("*.csv")))
            for file in category_dir.glob("*.csv"):
                total_size_bytes += os.path.getsize(file)

# Convert to GB for display
total_size_mb = total_size_bytes / (1024 * 1024)
total_size_gb = total_size_mb / 1024

print(f"Total size of the new dataset: {total_size_gb:.2f} GB")
print("Number of replicated files per category:")
for category, count in replicated_counts.items():
    print(f"  - {category}: {count}")
print(f"Total number of files in the new dataset: {total_files}")
print("--------------------------------------------------")