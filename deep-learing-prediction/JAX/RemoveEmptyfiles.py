import pandas as pd
import os


# you should  replace this path with the ROOT folder where your dataset is located etc. folders are located.
dataset_path = r"C:\Users\13202\Downloads\brainwave_readings\processed_csv\processed"

# Columns to be dropped
# Verify these names exactly match the columns in your CSV files.
columns_to_drop = [
    'Accel Channel 0',
    'Accel Channel 1',
    'Accel Channel 2',
    'Not Used',
    'Digital Channel 0 (D11)',
    'Digital Channel 1 (D12)',
    'Analog Channel 1'
]


print(f"Starting column drop operation in: {dataset_path}")
total_files_processed = 0

# os.walk iterates through the root folder and all subfolders (classes)
for root, dirs, files in os.walk(dataset_path):
    for file_name in files:
        if file_name.endswith(".csv"):
            file_path = os.path.join(root, file_name)
            
            try:
                # Read CSV
                df = pd.read_csv(file_path)

                
                df.columns = [col.replace('"', '').replace("'", '').replace('\r','').strip() for col in df.columns]

                # Drop specified columns if they exist in the DataFrame
                columns_actually_dropped = [col for col in df.columns if col in columns_to_drop]
                
                df = df.drop(columns=columns_actually_dropped, errors='ignore')

                # Save back overwriting the original file
                df.to_csv(file_path, index=False)
                
                total_files_processed += 1
                
                
                if columns_actually_dropped:
                    print(f"Processed {file_name} in {os.path.basename(root)}. Dropped: {len(columns_actually_dropped)} columns.")
                
            except Exception as e:
                print(f" ERROR: Could not process {file_path}. Reason: {e}")

print(f"\nFinished cleaning. Total files processed: {total_files_processed}.")