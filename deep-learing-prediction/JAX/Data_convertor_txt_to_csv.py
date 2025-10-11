import pandas as pd
import glob
import os


# Base directory where .txt files are
base_directory = r"C:\Users\13202\Downloads\brainwave_readings\brainwave_readings"
# The new directory where converted files will be saved
output_base_directory = r"C:\Users\13202\Downloads\brainwave_readings\processed_csv"


# Find all .txt files recursively
all_txt_files = glob.glob(os.path.join(base_directory, "**", "*.txt"), recursive=True)

print(f" Found {len(all_txt_files)} TXT files")

for txt_file in all_txt_files:
    try:
        print(f"\n Processing: {txt_file}")

        # Skip metadata lines starting with '%'
        df = pd.read_csv(txt_file, comment="%", delimiter=",")

        # Get the path relative to the base directory
        relative_path = os.path.relpath(os.path.dirname(txt_file), base_directory)

        # Construct the new output directory path
        output_directory = os.path.join(output_base_directory, relative_path)

        # Create the new directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Build the full path for the new .csv file
        csv_filename = os.path.splitext(os.path.basename(txt_file))[0] + ".csv"
        csv_file = os.path.join(output_directory, csv_filename)

        # Save as CSV in the new directory
        df.to_csv(csv_file, index=False)

        print(f" Converted {txt_file} → {csv_file}")
        print(" Preview:")
        print(df.head())

    except Exception as e:
        print(f" Error converting {txt_file}: {e}")

