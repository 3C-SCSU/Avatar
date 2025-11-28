#!/bin/sh
# Authors: Brady Theisen
# Fixed by: Arsalan Issue #333 - File Shuffler Unification Bug Fix
# Description: Fully POSIX-compliant version for cross-platform compatibility, 
#              with robust path handling for files containing spaces.

set -e # Exit on any error (kept commented out for debugging simplicity)
set -u # Exit on undefined variables

# Define categories (Space-separated string for POSIX compliance)
categories="backward down forward land landing left right steady takeoff TakeOff Takeoff turnleft turnright up"

TRUE_PATH="$1"
echo "bash.sh received path: \"$TRUE_PATH\""

# Validate input path
if [ ! -d "$TRUE_PATH" ]; then
    echo "ERROR: Directory '$TRUE_PATH' does not exist or is not accessible"
    exit 1
fi

root_dir="$TRUE_PATH"
echo "Root directory received: \"$root_dir\""

# Detect operating system for cross-platform compatibility
OS="$(uname -s)"
case "$OS" in
    Darwin*) STAT_CMD="stat -f %Sm" ;;   # macOS
    Linux*)  STAT_CMD="stat -c %y" ;;    # Linux
    *)       STAT_CMD="stat -c %y" ;;    # Default to Linux format
esac

total_categories_processed=0
total_files_processed=0
categories_with_issues_count=0
categories_with_issues="" 

echo "Starting file shuffling process..."
echo "Detected OS: $OS"

# Process each category
for category in $categories; do
    echo "Searching for category directory: \"$category\""
    
    category_path=$(find "$root_dir" -type d -name "$category" | head -n 1)
    
    if [ -z "$category_path" ]; then
        echo "No directory found for category: \"$category\""
        continue
    fi
    
    echo "Processing directory: \"$category_path\""
    total_categories_processed=$((total_categories_processed + 1))
    
    # Move all files from subdirectories to main category directory and remove subdirectories
    echo "Moving files from subdirectories to main category directory..."
    if ! find "$category_path" -mindepth 2 -type f -exec mv "{}" "$category_path" \; 2>/dev/null; then
        echo "Note: Some files may have already been in the main directory"
    fi
    
    echo "Removing empty subdirectories..."
    find "$category_path" -mindepth 1 -type d -exec rm -r "{}" \; 2>/dev/null || true
    
    file_count=$(find "$category_path" -maxdepth 1 -type f | wc -l | tr -d '[:space:]')
    echo "Found $file_count files to process in category: $category"
    
    if [ "$file_count" -eq 0 ]; then
        echo "No files found in category: $category, skipping..."
        continue
    fi
    
    echo "Renaming files to temporary names to avoid conflicts..."
    count=1
    rename_success_count=0
    
    temp_file_list=$(mktemp)

    find "$category_path" -maxdepth 1 -type f -print > "$temp_file_list"
    
    while IFS= read -r file; do
        filename=$(basename "$file") 
        # Get extension
        ext="${filename##*.}" 
        dir=$(dirname "$file")
        
        new_name="$dir/temp$count.$ext"
        
        echo "Renaming: \"$filename\" -> \"$(basename "$new_name")\""
        
        if mv "$file" "$new_name"; then
            rename_success_count=$((rename_success_count + 1))
            count=$((count + 1))
        else
            echo "ERROR: Failed to rename \"$file\" to \"$new_name\"" >&2
            categories_with_issues="$categories_with_issues $category"
            categories_with_issues_count=$((categories_with_issues_count + 1))
        fi
    done < "$temp_file_list"
    
    rm -f "$temp_file_list"

    echo "Successfully renamed $rename_success_count files with temporary names"
    
    if [ "$rename_success_count" -ne "$file_count" ]; then
        echo "WARNING: Rename count mismatch in category '$category'"
        categories_with_issues="$categories_with_issues $category"
        categories_with_issues_count=$((categories_with_issues_count + 1))
    fi
    
    echo "Randomizing filenames..."
    randomize_success_count=0
    
    temp_file_list_temp=$(mktemp)
    find "$category_path" -maxdepth 1 -type f -name 'temp*.*' -print > "$temp_file_list_temp"

    while IFS= read -r file; do
        random_number=$(awk 'BEGIN{srand(); print int(rand()*1000000)}')
        
        filename=$(basename "$file")
        ext="${filename##*.}"
        dir=$(dirname "$file")
        target_name="$dir/$category-$random_number.$ext"
        
        i=0
        while [ -f "$target_name" ]; do
            random_number=$(awk 'BEGIN{srand(); print int(rand()*1000000)}')
            target_name="$dir/$category-$random_number-$i.$ext"
            i=$((i + 1))
        done
        
        if mv "$file" "$target_name"; then
            randomize_success_count=$((randomize_success_count + 1))
        else
            echo "ERROR: Failed to randomize filename for \"$file\"" >&2
            categories_with_issues="$categories_with_issues $category"
            categories_with_issues_count=$((categories_with_issues_count + 1))
        fi
    done < "$temp_file_list_temp"
    
    rm -f "$temp_file_list_temp"
    echo "Successfully randomized $randomize_success_count filenames"
    
    echo "Randomizing timestamps..."
    timestamp_success_count=0
    
    temp_file_list_final=$(mktemp)
    find "$category_path" -maxdepth 1 -type f -name "$category-*.*" -print > "$temp_file_list_final"

    while IFS= read -r file; do
        if [ "$OS" = "Darwin" ]; then
            rand_time=$(date -v-"$((RANDOM % 10))"d -v"$((RANDOM % 24))"H -v"$((RANDOM % 60))"M +"%Y%m%d%H%M" 2>/dev/null)
        else
            rand_days=$((RANDOM % 10))
            rand_hours=$((RANDOM % 24))
            rand_minutes=$((RANDOM % 60))
            rand_time=$(date -d "$rand_days days ago $rand_hours hour $rand_minutes minute" +"%Y%m%d%H%M" 2>/dev/null)
        fi
        if [ -n "$rand_time" ] && touch -t "$rand_time" "$file" 2>/dev/null; then
            timestamp_success_count=$((timestamp_success_count + 1))
        else
            echo "WARNING: Failed to modify timestamp for \"$(basename "$file")\""
        fi
    done < "$temp_file_list_final"
    
    rm -f "$temp_file_list_final"

    final_file_count=$(find "$category_path" -maxdepth 1 -type f | wc -l | tr -d '[:space:]')
    total_files_processed=$((total_files_processed + final_file_count))
    
    echo "Category '$category' processing summary:"
    echo " Files found: $file_count"
    echo " Files renamed: $rename_success_count"
    echo " Files randomized: $randomize_success_count"
    echo " Timestamps modified: $timestamp_success_count"
    echo " Final file count: $final_file_count"
    
    if [ "$file_count" -ne "$final_file_count" ]; then
        echo "ERROR: File count mismatch in category '$category'!"
        categories_with_issues="$categories_with_issues $category"
        categories_with_issues_count=$((categories_with_issues_count + 1))
    else
        echo "SUCCESS: All files in category '$category' processed correctly"
    fi
done

echo "File shuffling process completed!"
echo "Categories processed: $total_categories_processed"
echo "Total files processed: $total_files_processed"

if [ "$categories_with_issues_count" -gt 0 ]; then
    echo "Categories with issues: $categories_with_issues"
    echo "WARNING: Some categories had processing issues"
    exit 1
else
    echo "SUCCESS: All categories processed without issues"
fi
