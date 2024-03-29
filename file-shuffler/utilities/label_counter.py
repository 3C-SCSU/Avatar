import pandas as pd
import os
import sys

# INFO:
# This script will count the total number of columns in each csv file in a directory
# Argument 1: The directory of files to process. Defaults to "./"

column_count = {}
csv_count = 0
non_csv_count = 0

# Get folderpath from commandline
folderpath = None
if sys.argv[1] == None: 
    folderpath = "./"
    print ("Using script's current directory")
else:
    folderpath = sys.argv[1]
    print ("Using " + folderpath)

def count_columns(filepath):
    try:
        df = pd.read_csv(file_path)
        for col in df.columns:
            # print (col)
            if col in column_count:
                column_count[col] += 1
            else:
                column_count[col] = 1

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Start processing files at folderpath
# Walking through the directory and its subdirectories
for dirpath, dirnames, files in os.walk(folderpath):

    foldername = os.path.basename(dirpath)

    # Check if there are any csv files in the folderpath
    csv_files = [f for f in files if f.endswith('.csv')]
    non_csv_files = [f for f in files if not f.endswith('.csv')]
    csv_count += len(csv_files)
    non_csv_count += len(non_csv_files)
    if csv_files:
        print(f"Processing folder: {foldername}")
        for file_name in csv_files:
            file_path = os.path.join(dirpath, file_name)
            count_columns(file_path)
        print(f"Finished processing folder: {foldername}")
        print(f"Processed {len(csv_files)} files")
    if non_csv_files:
        print (f"+++Found {len(non_csv_files)} non-CSV files+++")
        for file_name in non_csv_files:
            print (file_name)
        print ("+++++++++++++++++")

# Print results
print ("==========================")
for key, value in column_count.items():
    print (f"{key}: {value}")
print ("==========================")
print (f"Total files found: {csv_count+non_csv_count}")
print (f"CSV files processed: {csv_count}")
print (f"Non-CSV files skipped: {non_csv_count}")
print ("==========================")