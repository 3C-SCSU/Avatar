#!/usr/bin/bash
#Authors: Brady Theisen

TRUE_PATH="$1"

if [[ -z "$TRUE_PATH" ]]; then
  echo "No path provided. Usage: ./run.sh <path>"
  exit 1
fi

echo "Path passed from Python: $TRUE_PATH"

#covert the windows path to a linux path if need it
if [[ "$TRUE_PATH" =~ ^[a-zA-Z]:\\ ]]; then
  # Convert Windows path to WSL path
  DRIVE_LETTER=$(echo "$TRUE_PATH" | cut -c1 | tr 'A-Z' 'a-z')
  PATH_REMAINDER=$(echo "$TRUE_PATH" | cut -c3- | tr '\\' '/')
  TRUE_PATH="/mnt/$DRIVE_LETTER/$PATH_REMAINDER"
fi
echo "Converted path: $TRUE_PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#find pyton
if py --version &> /dev/null; then
  python_cmd="py"
elif python3 --version &> /dev/null; then
  python_cmd="python3"
elif python --version &> /dev/null; then
  python_cmd="python"
else
  echo "Python is not installed or not found in PATH."
  echo "please install python first"
  exit 1
fi

echo "Found python command: $python_cmd"
echo "Running python script"
"$python_cmd" "$SCRIPT_DIR/python.py" "$TRUE_PATH"
echo "Finished running python script"
echo "Running bash script"
bash "$SCRIPT_DIR/bash.sh" "$TRUE_PATH"
echo "Finished running bash script"


