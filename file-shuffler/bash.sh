#!/usr/bin/bash
# Authors: Brady Theisen
# Fixed by: Arsalan Issue #333 - File Shuffler Unification Bug Fix
# Description: Fixes subshell variable scope issue that caused file overwrites during renaming

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Define categories
categories=("backward" "down" "forward" "land" "landing" "left" "right" "steady" "takeoff" "TakeOff" "Takeoff" "turnleft" "turnright" "up")

TRUE_PATH="$1"
echo "bash.sh received path: \"$TRUE_PATH\""

# Validate input path
if [[ ! -d "$TRUE_PATH" ]]; then
    echo "ERROR: Directory '$TRUE_PATH' does not exist or is not accessible"
    exit 1
fi

root_dir="$TRUE_PATH"
echo "Root directory received: \"$root_dir\""

# Detect operating system for cross-platform compatibility
OS="$(uname -s)"
case "$OS" in
    Darwin*) STAT_CMD="stat -f %Sm" ;;  # macOS
    Linux*)  STAT_CMD="stat -c %y" ;;   # Linux
    *)       STAT_CMD="stat -c %y" ;;   # Default to Linux format
esac

# Track overall statistics
total_categories_processed=0
total_files_processed=0
categories_with_issues=()

echo "Starting file shuffling process..."
echo "Detected OS: $OS"

# Process each category
for category in "${categories[@]}"; do
    echo "Searching for category directory: \"$category\""
    
    # Find the first matching directory for this category
    category_path=$(find "$root_dir" -type d -name "$category" | head -n 1)
    
    if [[ -z "$category_path" ]]; then
        echo "No directory found for category: \"$category\""
        continue
    fi
    
    echo "Processing directory: \"$category_path\""
    ((total_categories_processed++))
    
    # Move all files to the category directory and delete subdirectories
    echo "Moving files from subdirectories to main category directory..."
    if ! find "$category_path" -mindepth 2 -type f -exec mv "{}" "$category_path" \; 2>/dev/null; then
        echo "Note: Some files may have already been in the main directory"
    fi
    
    echo "Removing empty subdirectories..."
    find "$category_path" -mindepth 1 -type d -exec rm -r "{}" \; 2>/dev/null || true
    
    # Count files before processing for verification
    file_count=$(find "$category_path" -maxdepth 1 -type f | wc -l)
    echo "Found $file_count files to process in category: $category"
    
    if [[ "$file_count" -eq 0 ]]; then
        echo "No files found in category: $category, skipping..."
        continue
    fi
    
    # CORE FIX: Rename files to avoid overwrite using process substitution
    echo "Renaming files to temporary names to avoid conflicts..."
    count=1
    rename_success_count=0
    
    while IFS= read -r file; do
        if [[ -f "$file" ]]; then
            new_name="${file%/*}/temp$count.${file##*.}"
            echo "Renaming: \"$(basename "$file")\" -> \"$(basename "$new_name")\""
            
            if mv "$file" "$new_name" 2>/dev/null; then
                ((rename_success_count++))
                ((count++))
            else
                echo "ERROR: Failed to rename \"$file\""
                categories_with_issues+=("$category")
            fi
        fi
    done < <(find "$category_path" -maxdepth 1 -type f)
    
    echo "Successfully renamed $rename_success_count files with temporary names"
    
    if [[ "$rename_success_count" -ne "$file_count" ]]; then
        echo "WARNING: Rename count mismatch in category '$category'"
        categories_with_issues+=("$category")
    fi
    
    # Randomize file names
    echo "Randomizing filenames..."
    files=($(find "$category_path" -maxdepth 1 -type f))
    num_files=${#files[@]}
    
    if [[ "$num_files" -eq 0 ]]; then
        echo "ERROR: No files found after renaming in category: $category"
        categories_with_issues+=("$category")
        continue
    fi
    
    randomize_success_count=0
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            # Generate unique random number
            random_number=$(( RANDOM % num_files + 1 ))
            target_name="${file%/*}/$random_number.${file##*.}"
            
            # Ensure uniqueness
            while [[ -f "$target_name" ]]; do
                random_number=$(( RANDOM % num_files + 1 ))
                target_name="${file%/*}/$random_number.${file##*.}"
            done
            
            if mv "$file" "$target_name" 2>/dev/null; then
                ((randomize_success_count++))
            else
                echo "ERROR: Failed to randomize filename for \"$file\""
                categories_with_issues+=("$category")
            fi
        fi
    done
    
    echo "Successfully randomized $randomize_success_count filenames"
    
    # Change timestamps with cross-platform support
    echo "Randomizing timestamps..."
    timestamp_success_count=0
    
    for file in "$category_path"/*; do
        if [[ -f "$file" ]]; then
            # Generate random time (0-9 days ago)
            if [[ "$OS" == "Darwin" ]]; then
                # macOS date command
                rand_time=$(date -v-"$((RANDOM % 10))"d -v"$((RANDOM % 24))"H -v"$((RANDOM % 60))"M +"%Y%m%d%H%M" 2>/dev/null)
            else
                # Linux date command
                rand_time=$(date -d "$((RANDOM % 10)) days ago $((RANDOM % 24)) hour $((RANDOM % 60)) minute" +"%Y%m%d%H%M" 2>/dev/null)
            fi
            
            if [[ -n "$rand_time" ]] && touch -t "$rand_time" "$file" 2>/dev/null; then
                ((timestamp_success_count++))
            else
                echo "WARNING: Failed to modify timestamp for \"$(basename "$file")\""
            fi
        fi
    done
    
    # Final verification
    final_file_count=$(find "$category_path" -maxdepth 1 -type f | wc -l)
    ((total_files_processed += final_file_count))
    
    echo "Category '$category' processing summary:"
    echo " Files found: $file_count"
    echo " Files renamed: $rename_success_count"  
    echo " Files randomized: $randomize_success_count"
    echo " Timestamps modified: $timestamp_success_count"
    echo " Final file count: $final_file_count"
    
    if [[ "$file_count" -ne "$final_file_count" ]]; then
        echo "ERROR: File count mismatch in category '$category'!"
        categories_with_issues+=("$category")
    else
        echo "SUCCESS: All files in category '$category' processed correctly"
    fi
done

# Final summary
echo "File shuffling process completed!"
echo "Categories processed: $total_categories_processed"
echo "Total files processed: $total_files_processed"

if [[ ${#categories_with_issues[@]} -gt 0 ]]; then
    echo "Categories with issues: ${categories_with_issues[*]}"
    echo "WARNING: Some categories had processing issues"
    exit 1
else
    echo "SUCCESS: All categories processed without issues"
fi