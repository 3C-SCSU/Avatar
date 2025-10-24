# Authors: Brady Theisen
import os
import sys

import pandas as pd


def convert_txt_to_csv(file_path):
    try:
        # Skip the first four header lines
        df = pd.read_csv(file_path, sep=",", skiprows=4, on_bad_lines="skip")

        # Convert to CSV format and save
        csv_file_path = file_path.rsplit(".", 1)[0] + ".csv"
        df.to_csv(csv_file_path, index=False)

        # Remove original txt file
        os.remove(file_path)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def main(directory):
    # Walking through the directory and its subdirectories
    for dirpath, dirnames, files in os.walk(directory):
        category = os.path.basename(dirpath)

        # Check if there are any txt files in the directory
        txt_files = [f for f in files if f.endswith(".txt")]
        if txt_files:
            print(f"Processing category: {category}")
            for file_name in txt_files:
                file_path = os.path.join(dirpath, file_name)
                convert_txt_to_csv(file_path)
            print(f"Finished processing category: {category}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python python.py <directory>")
        sys.exit(1)
    main(sys.argv[1])
