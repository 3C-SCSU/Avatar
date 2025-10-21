#!/usr/bin/bash
# Authors: Brady Theisen

TRUE_PATH="$1"

if [[ -z "$TRUE_PATH" ]]; then
    echo "ERROR: No directory path argument provided."
    echo "Usage: $0 <directory_path>"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Path passed from Python: $TRUE_PATH"

echo "Scanning for valid Python executables..."

# Find all python executables in PATH and standard locations, filter only actual executables
PYTHON_EXES=$(find "$HOME/AppData/Local/Programs/Python" "/usr/bin" "/usr/local/bin" "/c/Python*" "/c/Users/$(whoami)/AppData/Local/Programs/Python" 2>/dev/null \
    -type f \( -iname "python.exe" -o -iname "python3.exe" -o -iname "python" -o -iname "python3" \) | sort -u)

# Add python3 and python from PATH
for exe in python3 python; do
    p=$(command -v $exe 2>/dev/null)
    [[ -n "$p" ]] && PYTHON_EXES="$PYTHON_EXES $p"
done

# Choose highest version Python with required modules
get_py_version() { "$1" --version 2>&1 | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "0.0.0"; }
REQUIRED_MODULE=pandas
SELECTED_PY=""
BEST_VER="0.0.0"

for py in $PYTHON_EXES; do
    [[ "$py" =~ pythonw\.exe$ ]] && continue
    [[ ! -x "$py" ]] && continue
    version=$(get_py_version "$py")
    # Skip Microsoft Store aliases by checking version output
    store_check=$("$py" --version 2>&1 | grep -qi "Microsoft Store"; echo $?)
    [[ "$store_check" -eq 0 ]] && continue
    # Check required module
    "$py" -c "import $REQUIRED_MODULE" 2>/dev/null || continue
    if [[ "$version" > "$BEST_VER" ]]; then
        SELECTED_PY="$py"
        BEST_VER="$version"
    fi
done

if [[ -z "$SELECTED_PY" ]]; then
    echo "ERROR: No valid Python found with '$REQUIRED_MODULE'. Please install Python 3 and 'pip install pandas'."
    exit 1
fi

echo "Using Python at $SELECTED_PY (version $BEST_VER)"
echo "Command to run: $SELECTED_PY $SCRIPT_DIR/python.py $TRUE_PATH"

"$SELECTED_PY" "$SCRIPT_DIR/python.py" "$TRUE_PATH"
echo "Finished running python script"

echo "Running bash script"
bash "$SCRIPT_DIR/bash.sh" "$TRUE_PATH"
echo "Finished running bash script"
