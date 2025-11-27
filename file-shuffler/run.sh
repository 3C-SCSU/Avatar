#!/bin/sh

# Authors: Based on Brady Theisen's original code, updated for POSIX compliance.
# Purpose: Robustly finds and executes the python.py script across various Unix-like systems.
# This version prioritizes the Python path passed by run_file_shuffler.py.

# --- Configuration ---
REQUIRED_MODULE="pandas"
PYTHON_SCRIPT="python.py"
BASH_SCRIPT="bash.sh"
# ---------------------

TRUE_PATH="$1"  # Argument 1: Directory path for processing
PY_EXE="$2"     # Argument 2: Python executable path (passed from run_file_shuffler.py)

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# 1. Input Validation
if [ -z "$TRUE_PATH" ]; then
  echo "ERROR: No directory path argument provided." >&2
  echo "Usage: $0 <directory_path> [python_executable]"
  exit 1
fi

# 2. Function to Find Python Executable (Fallback only)
find_python() {
  echo "Scanning for valid Python executables in PATH (Fallback Mode)..."
  
  # Search for python3, then python, prioritizing those in PATH
  PYTHON_EXES="$(command -v python3 2>/dev/null) $(command -v python 2>/dev/null)"
  
  for py in $PYTHON_EXES; do
    if [ -x "$py" ]; then
      # Check for required module: 'pandas'
      "$py" -c "import $REQUIRED_MODULE" 2>/dev/null
      if [ $? -eq 0 ]; then
        echo "$py" # Output the selected Python executable path
        return 0
      fi
    fi
  done
  
  echo "ERROR: No valid Python found with '$REQUIRED_MODULE' in PATH. Please install Python 3 and 'pip install pandas'." >&2
  return 1
}

# 3. Determine Python Executable
if [ -n "$PY_EXE" ]; then
  # PRIORITY: Use the path passed from the QML application (PY_EXE = $2)
  SELECTED_PY="$PY_EXE"
  echo "Using Python (from argument): $SELECTED_PY"
  
  # Quick validation to ensure the passed executable has 'pandas'
  # This uses the specific executable provided by run_file_shuffler.py
  if ! "$SELECTED_PY" -c "import $REQUIRED_MODULE" 2>/dev/null; then
    echo "ERROR: Python executable '$SELECTED_PY' does not have '$REQUIRED_MODULE'." >&2
    exit 1
  fi
  
else
  # Fallback: Search PATH if no argument was provided
  SELECTED_PY=$(find_python)
  if [ $? -ne 0 ]; then
    exit 1 # find_python failed and printed an error
  fi
  echo "Using found Python: $SELECTED_PY"
fi

# 4. Run Python Script
FULL_PYTHON_SCRIPT="$SCRIPT_DIR/$PYTHON_SCRIPT"
echo "Command to run: $SELECTED_PY $FULL_PYTHON_SCRIPT $TRUE_PATH"

if [ ! -f "$FULL_PYTHON_SCRIPT" ]; then
  echo "ERROR: Python script not found at $FULL_PYTHON_SCRIPT" >&2
  exit 1
fi

# Execute the python script with the selected interpreter
"$SELECTED_PY" "$FULL_PYTHON_SCRIPT" "$TRUE_PATH"

if [ $? -ne 0 ]; then
  echo "WARNING: Python script finished with errors." >&2
fi
echo "Finished running python script"

# 5. Run Bash Script (if it exists)
FULL_BASH_SCRIPT="$SCRIPT_DIR/$BASH_SCRIPT"

if [ ! -f "$FULL_BASH_SCRIPT" ]; then
  echo "WARNING: Bash script not found at $FULL_BASH_SCRIPT. Skipping."
else
  # Use 'sh' explicitly to run the POSIX-compliant bash.sh file
  sh "$FULL_BASH_SCRIPT" "$TRUE_PATH"
  if [ $? -ne 0 ]; then
    echo "WARNING: Bash script finished with errors." >&2
  fi
fi

echo "Finished running shell script execution chain."
exit 0
