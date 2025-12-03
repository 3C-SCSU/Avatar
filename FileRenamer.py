import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


CATEGORIES = [
    #directories need to be adjusted if used
    r"C:\Users\ayomi\Downloads\rawdata_spring2025-20251004T004727Z-1-001\processed\right",
    r"C:\Users\ayomi\Downloads\rawdata_spring2025-20251004T004727Z-1-001\processed\left",
    r"C:\Users\ayomi\Downloads\rawdata_spring2025-20251004T004727Z-1-001\processed\forward",
    r"C:\Users\ayomi\Downloads\rawdata_spring2025-20251004T004727Z-1-001\processed\backward",
    r"C:\Users\ayomi\Downloads\rawdata_spring2025-20251004T004727Z-1-001\processed\landing",
    r"C:\Users\ayomi\Downloads\rawdata_spring2025-20251004T004727Z-1-001\processed\takeoff",
]

VALID_EXTENSIONS = {".csv", ".txt"}



def rename_initial_files(directory):
    files = sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    )

    if not files:
        return

    # First convert everything to temporary names
    temp_list = []
    for idx, f in enumerate(files, start=1):
        old_path = os.path.join(directory, f)
        _, ext = os.path.splitext(f)
        temp = f"TEMP_{idx:04d}{ext}"
        temp_path = os.path.join(directory, temp)
        os.rename(old_path, temp_path)
        temp_list.append(temp)

    # Now rename sequentially
    for idx, temp_file in enumerate(temp_list, start=1):
        old_path = os.path.join(directory, temp_file)
        _, ext = os.path.splitext(temp_file)
        new_path = os.path.join(directory, f"{idx}{ext}")
        os.rename(old_path, new_path)

    print(f"[INITIAL RENAME] {directory} has been cleaned and sequenced.")


def rename_new_file(directory, new_file):
    time.sleep(0.5)  # Ensure file is fully written

    # Ignore invalid extensions
    _, ext = os.path.splitext(new_file)
    if ext.lower() not in VALID_EXTENSIONS:
        return

    files = [
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    ]

    # Determine next index
    next_index = len(files)

    old_path = os.path.join(directory, new_file)
    new_path = os.path.join(directory, f"{next_index}{ext}")

    # Prevent overwriting by incrementing until unique
    while os.path.exists(new_path):
        next_index += 1
        new_path = os.path.join(directory, f"{next_index}{ext}")

    os.rename(old_path, new_path)
    print(f"[NEW FILE RENAMED] {new_file} → {next_index}{ext}")



class CategoryHandler(FileSystemEventHandler):

    def __init__(self, directory):
        self.directory = directory

    def on_created(self, event):
        if not event.is_directory:
            filename = os.path.basename(event.src_path)
            print(f"[DETECTED] New file in {self.directory}: {filename}")
            rename_new_file(self.directory, filename)



def start_monitor():
    observers = []

    for directory in CATEGORIES:
        print(f"[STARTUP] Cleaning and sequencing: {directory}")
        rename_initial_files(directory)     # FIXED: Only once on startup

        observer = Observer()
        observer.schedule(CategoryHandler(directory), directory, recursive=False)
        observer.start()
        observers.append(observer)

    print("\n[READY] Watching all folders...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for o in observers:
            o.stop()
        for o in observers:
            o.join()


if __name__ == "__main__":
    start_monitor()

