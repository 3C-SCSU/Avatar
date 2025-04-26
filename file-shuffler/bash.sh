#!/usr/bin/bash

# Authors: Brady Theisen

# Define categories
categories=("backward" "down" "forward" "land" "left" "right" "steady" "takeoff" "turnleft" "turnright" "up")

# Define the root to search from (can be current directory or elsewhere)
root_dir="../"  # or just "." if you're running from project root

# Process each category
for category in "${categories[@]}"; do
    echo "Searching for category directory: $category"

    # Find the first matching directory for this category
    category_path=$(find "$root_dir" -type d -name "$category" | head -n 1)

    if [ -z "$category_path" ]; then
        echo "No directory found for category: $category"
        continue
    fi

    echo "Processing directory: $category_path"

    #Move all files to the category directory and delete subdirectories
    find "$category_path" -mindepth 2 -type f -exec mv {} "$category_path" \;
    find "$category_path" -mindepth 1 -type d -exec rm -r {} \;

    # Rename files to avoid overwrite
    files=($(find "$category_path" -maxdepth 1 -type f))
    count=1
    find "$category_path" -maxdepth 1 -type f | while IFS= read -r file; do
      new_name="${file%/*}/temp$count.${file##*.}"
      echo "Renaming:"
      echo "  Old name: $file"
      mv "$file" "$new_name"
      echo "  New name: $new_name"
      #echo "------------------------------"
    ((count++))
done

    # Randomize file names
    files=($(find "$category_path" -maxdepth 1 -type f))
    num_files=${#files[@]}
    for file in "${files[@]}"; do
        random_number=$(( RANDOM % num_files + 1 ))
        while test -f "${file%/*}/$random_number.${file##*.}"; do
            random_number=$(( RANDOM % num_files + 1 ))
        done
        mv "$file" "${file%/*}/$random_number.${file##*.}"
    done

    # Change timestamps
    for file in "$category_path"/*; do
        original_timestamp=$(stat -c %y "$file")
        echo "Original timestamp of $file: $original_timestamp"

        rand_time=$(date -d "$((RANDOM % 10)) days ago $((RANDOM % 24)) hour $((RANDOM % 60)) minute" +"%Y%m%d%H%M")
        touch -t "$rand_time" "$file"

        modified_timestamp=$(stat -c %y "$file")
        echo "Modified timestamp of $file: $modified_timestamp"
    done
done
