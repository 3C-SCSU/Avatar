# Determines the user's current operating system and runs run.sh based on which OS they're on

import os
import platform
import subprocess

def find_windows_sh_exe_location():
    common_dirs = [
        os.getenv('SystemRoot', 'C:\\Windows') + '\\System32',
        'C:\\Program Files',
        'C:\\Program Files (x86)'
    ]
    
    def search_directory(directory):
        for root, dirs, files in os.walk(directory):
            if 'sh.exe' in files:
                return os.path.join(root, 'sh.exe')
        return None

    for directory in common_dirs:
        result = search_directory(directory)
        if result:
            return result

    try:
        output = subprocess.check_output('where sh.exe', shell=True, text=True)
        return output.strip().split('\n')[0]  # Return the first found path
    except subprocess.CalledProcessError:
        pass  # Handle the case where `where` command fails

    return None

def convert_to_universal_path(windows_path):
    # Convert backslashes to forward slashes
    universal_path = windows_path.replace('\\', '/')
    
    # Wrap the path with quotes if it contains spaces
    if ' ' in universal_path:
        universal_path = f'"{universal_path}"'
    
    return universal_path

def main():
    current_os = platform.system()
    if current_os == "Windows":
        print("Finding sh.exe to run the script...")
        sh_exe_path = find_windows_sh_exe_location()

        if(len(sh_exe_path) > 0):
            print(f"\nFound sh.exe at {sh_exe_path}, running the file shuffler... \n")

            print("File shuffler output:")
            sh_exe_path = convert_to_universal_path(sh_exe_path)
        else:
            print("Failed to find sh.exe path, ensure that sh.exe is installed (check git bash installation)")
        
        os.system(f"{sh_exe_path} ./run.sh")
    elif current_os == "Linux":
        os.system("sh ./run.sh")
    elif current_os == "Darwin":  # macOS
        os.system("sh ./run.sh")
    else:
        print(f"{current_os} detected - attempting `sh ./run.sh`")

main()