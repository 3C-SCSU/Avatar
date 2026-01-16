import sys
import os
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot
import io
import urllib.parse
import contextlib
import shutil
import json

# Add file-opendata to Python path
sys.path.append(str(Path(__file__).resolve().parent / "file-opendata"))

class OpenDataAPI(QObject):
    """
    API wrapper for opendata.py that provides QML integration.
    Handles directory validation and dataset augmentation execution.
    """

    def __init__(self):
        super().__init__()

    @Slot(str, result=str)
    def validate_directory(self, base_dir):
        """
        Validates that the selected directory contains a brainwaves/ subdirectory
        with the expected structure.

        Args:
            base_dir: Path to parent directory (should contain brainwaves/)

        Returns:
            JSON string with validation result:
            - {"valid": true, "files": N, "estimated_new": M, "categories": K}
            - {"valid": false, "error": "error message"}
        """
        # Decode URL path (handle file:// prefix from QML FolderDialog)
        if base_dir.startswith("file:///"):
            base_dir = urllib.parse.unquote(base_dir.replace("file://", ""))
            if os.name == 'nt' and base_dir.startswith("/"):
                base_dir = base_dir[1:]  # Remove leading slash on Windows

        try:
            base_path = Path(base_dir)
            brainwaves_dir = base_path / "brainwaves"

            # Check if brainwaves directory exists
            if not brainwaves_dir.exists():
                return json.dumps({"valid": False, "error": "No brainwaves/ subdirectory found"})

            # Check if brainwaves has subdirectories
            subdirs = [d for d in brainwaves_dir.iterdir() if d.is_dir()]
            if not subdirs:
                return json.dumps({"valid": False, "error": "brainwaves/ directory is empty"})

            # Check for expected categories in at least one subdirectory
            expected_categories = ['forward', 'backward', 'land', 'takeoff', 'right', 'left']
            found_categories = set()

            for subdir in subdirs:
                for category_dir in subdir.iterdir():
                    if category_dir.is_dir() and category_dir.name in expected_categories:
                        found_categories.add(category_dir.name)

            if not found_categories:
                return json.dumps({"valid": False, "error": "No valid category folders found (forward, backward, land, takeoff, right, left)"})

            # Count total CSV files
            total_files = 0
            for subdir in subdirs:
                for category in expected_categories:
                    category_path = subdir / category
                    if category_path.exists():
                        csv_files = list(category_path.glob("*.csv"))
                        total_files += len(csv_files)

            if total_files == 0:
                return json.dumps({"valid": False, "error": "No CSV files found in category folders"})

            # Calculate estimated increase (60%)
            estimated_increase = int(total_files * 0.6)

            return json.dumps({
                "valid": True,
                "files": total_files,
                "estimated_new": estimated_increase,
                "categories": len(found_categories)
            })

        except Exception as e:
            return json.dumps({"valid": False, "error": f"Validation error: {str(e)}"})

    @Slot(str, result=str)
    def run_open_data(self, base_dir):
        """
        Runs the opendata.py script on the selected directory.
        Captures and returns all console output.

        Args:
            base_dir: Path to parent directory (contains brainwaves/)

        Returns:
            String containing all console output from opendata.py execution
        """
        # Decode URL path
        if base_dir.startswith("file:///"):
            base_dir = urllib.parse.unquote(base_dir.replace("file://", ""))
            if os.name == 'nt' and base_dir.startswith("/"):
                base_dir = base_dir[1:]

        print(f"Running Open Data augmentation on: {base_dir}")

        # Store original working directory
        original_cwd = os.getcwd()

        try:
            # Change to the file-opendata directory (where opendata.py expects to run from)
            # The script will look for brainwaves/ subdirectory from here
            opendata_dir = Path(__file__).resolve().parent / "file-opendata"
            os.chdir(base_dir)  # Run from the parent directory

            # Capture stdout and stderr
            output = io.StringIO()

            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                # Import and execute opendata.py
                script_path = opendata_dir / "opendata.py"

                # Read the script content
                with open(script_path, 'r') as f:
                    script_content = f.read()

                # Set up execution environment
                exec_globals = {
                    '__file__': str(script_path),
                    '__name__': '__main__',
                    'Path': Path,
                    'os': os,
                    'random': __import__('random'),
                    'shutil': shutil,
                    're': __import__('re')
                }

                try:
                    # Execute the script
                    exec(script_content, exec_globals)
                    print("\n✓ Open Data augmentation completed successfully!")

                except Exception as e:
                    print(f"\n✗ Error during augmentation: {str(e)}")
                    import traceback
                    print("\nFull traceback:")
                    print(traceback.format_exc())

            return output.getvalue()

        except FileNotFoundError as e:
            return f"Error: Directory or script not found - {str(e)}\n\nPlease ensure:\n- The selected directory exists\n- The brainwaves/ subdirectory is present\n- You have read access to the directory"

        except PermissionError as e:
            return f"Error: Permission denied - {str(e)}\n\nPlease ensure you have write access to:\n- The selected directory\n- The brainwaves/ subdirectory\n- All CSV files within category folders"

        except Exception as e:
            import traceback
            error_msg = f"Error during Open Data execution: {str(e)}\n\n"
            error_msg += "Please check:\n"
            error_msg += "- Directory contains brainwaves/ subdirectory\n"
            error_msg += "- CSV files exist in category folders (forward, backward, land, takeoff, right, left)\n"
            error_msg += "- You have write permissions for all files and directories\n"
            error_msg += "- Sufficient disk space is available\n\n"
            error_msg += f"Technical details:\n{traceback.format_exc()}"
            return error_msg

        finally:
            # Always return to original directory
            os.chdir(original_cwd)
