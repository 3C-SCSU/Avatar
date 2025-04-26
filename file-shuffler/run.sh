#!/usr/bin/bash
#Authors: Brady Theisen


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Running python script"
python_cmd=$(command -v python3 || command -v python)
"$python_cmd" "$SCRIPT_DIR/python.py"

echo "Finished running python script"

echo "Running bash script"
bash "$SCRIPT_DIR/bash.sh"
echo "Finished running bash script"
