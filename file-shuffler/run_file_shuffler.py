import os
import sys
import platform
import subprocess

def strip_file_prefix(p):
    if p.startswith('file:///'):
        p = p[8:] if len(p) > 8 and p[9:10] == ':' else p[7:]
    if os.name == 'nt' and p.startswith('/') and len(p) > 2 and p[2] == ':' and p[0] == '/':
        p = p[1:]
    return p

def find_windows_bash():
    possible_bashes = [
        "C:\\Program Files\\Git\\bin\\bash.exe",
        "C:\\Program Files (x86)\\Git\\bin\\bash.exe",
        "C:\\Windows\\System32\\bash.exe",
        "bash",
        "sh.exe"
    ]
    for bash in possible_bashes:
        try:
            subprocess.check_output(f'where {bash}', shell=True)
            return bash
        except subprocess.CalledProcessError:
            continue
    return None

def main(path):
    path = strip_file_prefix(path)
    print(f"Shuffling: {path}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_sh_location = os.path.join(script_dir, "run.sh")
    print("Here is the current file path:", script_dir)
    print(f"Looking for run.sh at: {run_sh_location}")

    if not os.path.exists(run_sh_location):
        error_msg = f"ERROR: run.sh not found at {run_sh_location}"
        print(error_msg)
        return error_msg

    normalized_path = ""
    if path:
        if os.path.isabs(path):
            normalized_path = os.path.normpath(path)
        else:
            normalized_path = os.path.abspath(os.path.normpath(path))
        if not os.path.exists(normalized_path):
            error_msg = f"ERROR: The directory {normalized_path} does not exist."
            print(error_msg)
            return error_msg
        os.chdir(normalized_path)
        print(f"Changed working directory to {normalized_path}")

    current_os = platform.system()
    output = ""
    python_exe = sys.executable  # Use the currently running Python interpreter

    if current_os == "Windows":
        bash_path = find_windows_bash()
        if not bash_path:
            error_msg = "ERROR: Bash shell not found on Windows. Please install Git Bash."
            print(error_msg)
            return error_msg
        print(f"Found bash/sh.exe at {bash_path}")

        cmd = [bash_path, run_sh_location, normalized_path, python_exe]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout + result.stderr

    elif current_os in ("Linux", "Darwin"):
        # Command structure: [sh, run.sh, TRUE_PATH, PYTHON_EXE]
        cmd = ["sh", run_sh_location, normalized_path, python_exe]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout + result.stderr

    else:
        print(f"{current_os} detected - attempting run.sh")
        # Command structure: [sh, run.sh, TRUE_PATH, PYTHON_EXE]
        cmd = ["sh", run_sh_location, normalized_path, python_exe]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout + result.stderr

    print(output)
    return output

if __name__ == "__main__":
    main("")
