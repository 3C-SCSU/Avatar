# This program was implemented by Ashhad Waquas Syed - 16019920
# As part of final exam completion for the course CSCI 610 taught by Dr.Cavalcanti

import datetime
import os
import random
import shutil
from collections import deque

import filedate
import schedule


def runScript():
    # Initializing path variables
    DATASET_PATH = "./data"

    OUTPUT_LOG_PATH = "./outputLog.txt"

    INTERMEDIATE_DATASET_PATH = "./data02"

    # New path for dataset
    NEW_DIRECTORY = "./data"

    # File operations to log the output
    FILE = open(OUTPUT_LOG_PATH, "a")

    FILE.write(
        f"\n\n-------THIS ITERATION WAS RAN ON: {datetime.datetime.now()}-------\n\n"
    )

    FILE.write("------------BEFORE MODIFICATION------------\n\n")

    for dirpath, dirname, filenames in os.walk(DATASET_PATH):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_metadata = filedate.File(file_path)
            old_date = file_metadata.get()
            # Logging file timestamp from metadata
            FILE.write(
                f"File {file_path} creation timestamp was {old_date['created']} before modification \n"
            )

    num_of_files = 0
    num_of_subdirs = 0
    dirnames = []
    for dirpath, dirname, filenames in os.walk(DATASET_PATH):
        if len(dirname) != 0:
            dirnames.append(dirname)
        if len(filenames) != 0:
            num_of_files += len(filenames)

    num_of_subdirs = len(dirnames[0])

    shutil.copytree(
        DATASET_PATH, INTERMEDIATE_DATASET_PATH, ignore=shutil.ignore_patterns("*.csv")
    )

    for i in range(num_of_subdirs):
        print(dirnames[0][i])
        # Initialize a list with random numbers within range of 1,num_of_files
        list_file_names = list(range(1, int(num_of_files / num_of_subdirs) + 1))

        new_file_names = deque(list_file_names)

        random.shuffle(new_file_names)

        for dirpath, dirname, filenames in os.walk(
            os.path.join(DATASET_PATH, dirnames[0][i])
        ):
            for f in filenames:
                name, ext = os.path.splitext(f)

                src = os.path.join(dirpath, f)
                dst = os.path.join(
                    INTERMEDIATE_DATASET_PATH,
                    dirnames[0][i],
                    f"{new_file_names.pop()}{ext}",
                )

                shutil.move(src, dst)

    shutil.rmtree(DATASET_PATH)
    os.rename(INTERMEDIATE_DATASET_PATH, NEW_DIRECTORY)

    FILE.write("\n\n------------AFTER MODIFICATION------------\n\n")

    for dirpath, dirname, filenames in os.walk(NEW_DIRECTORY):
        for f in filenames:
            rand_days = random.randint(10, 14)
            rand_mins = random.randint(1, 60)
            rand_secs = random.randint(1, 60)

            new_file_path = os.path.join(dirpath, f)

            rand_days = random.randint(10, 14)
            rand_mins = random.randint(1, 60)
            rand_secs = random.randint(1, 60)

            # Modifying file metadata
            new_file_date = str(
                datetime.datetime.now()
                - datetime.timedelta(
                    days=rand_days, seconds=rand_secs, minutes=rand_mins
                )
            )

            os.system(f"sudo touch -c -d {new_file_date} {new_file_path}")

            # Log new file metadata to ouput log
            FILE.write(
                f"File is now at {new_file_path}, creation time changed to : {new_file_date}\n"
            )

    FILE.write("\nIteration complete!\n")

    FILE.close()

    print(f"\n\nIteration complete! Check file at {OUTPUT_LOG_PATH} for information")
    print("\nPress Ctrl+C to abort this automated job. Thanks!")


runScript()


# Automate job to run every 30 seconds
schedule.every(30).seconds.do(runScript)

# Execute job queue
while True:
    schedule.run_pending()
