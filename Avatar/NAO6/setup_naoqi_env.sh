#!/bin/bash

# Adjust the path to where you have installed the NAOqi SDK to its lib or python subdirectory where the .so and .py modules are actually located.

# Inside your downloaded folder, you likely have either:
#   - lib/python2.7/site-packages/
#   - or directly .so files for naoqi.so / qi.so.


# Check if the script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[!] Please run me with 'source' instead of executing."
  exit 1
fi

NAOQI_SDK_PATH="$HOME/Downloads/pynaoqi-python2.7-2.8.7.4-linux64-20210819_141148/lib/python2.7/site-packages"
# NAOQI_SDK_PATH="/mnt/c/Users/91968/OneDrive/Desktop/Repos/naoqi-sdk/pynaoqi-python2.7-2.8.7.4-linux64-20210819_141148/lib/python2.7/site-packages"

if [ -d "$NAOQI_SDK_PATH" ]; then
  export PYTHONPATH="$PYTHONPATH:$NAOQI_SDK_PATH"
  echo "[+] PYTHONPATH updated successfully!"
  echo "Current PYTHONPATH $PYTHONPATH"
else
  echo "[!] Error: NAOqi SDK not found at $NAOQI_SDK_PATH"
  echo "Please provide correct SDK path."
fi
