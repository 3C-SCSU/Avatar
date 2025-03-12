#Group 8
#Cha Vue, Kevin Gutierrez, Sagar Neupane

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

def remove_quotes(input):
    return input.replace("\"", "")

def main(path):
    # Get the directory where run_file_shuffler.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Look for run.sh in the same directory as this script
    run_sh_location = os.path.join(script_dir, "run.sh")
    
    # Debug print to verify the path
    print(f"Looking for run.sh at: {run_sh_location}")
    
    if not os.path.exists(run_sh_location):
        print(f"ERROR: run.sh not found at {run_sh_location}")
        return f"ERROR: run.sh not found at {run_sh_location}"

    working_dir = os.getcwd()
    if(len(path) > 0):
        universalParameterizedPath = convert_to_universal_path(path)
        universalParameterizedPath = remove_quotes(universalParameterizedPath)
        os.chdir(universalParameterizedPath)
        print(f"Running file shuffler in {universalParameterizedPath}")

    current_os = platform.system()
    if current_os == "Windows":
        print("Finding sh.exe to run the script...")
        sh_exe_path = find_windows_sh_exe_location()

        if sh_exe_path and len(sh_exe_path) > 0:
            print(f"\nFound sh.exe at {sh_exe_path}, running the file shuffler... \n")

            print("File shuffler output:")
            sh_exe_path = convert_to_universal_path(sh_exe_path)
        else:
            print("Failed to find sh.exe path, ensure that sh.exe is installed (check git bash installation)")
            return "Failed to find sh.exe path, ensure that sh.exe is installed (check git bash installation)"
        
        run_sh_location = convert_to_universal_path(run_sh_location)

        print("Running file shuffler, please wait...\n")

        result = subprocess.run(
            ["powershell", f"& {sh_exe_path} {run_sh_location}"],
            shell=True,
            capture_output=True,
            text=True
        )
        
        # Change back to original working directory
        os.chdir(working_dir)
        
        # Combine stdout and stderr and filter only new lines
        output = result.stdout + result.stderr
        new_lines = [line for line in output.split('\n') if line]

        print('\n'.join(new_lines)) #print the output to the console

        return '\n'.join(new_lines) #return the output for the gui to consume

    elif current_os == "Linux":
        result = subprocess.run(
            ["sh", run_sh_location],  # Use the absolute path
            shell=True,
            capture_output=True,
            text=True
        )
        # Change back to original working directory
        os.chdir(working_dir)
        output = result.stdout + result.stderr
        return output
        
    elif current_os == "Darwin":  # macOS
        result = subprocess.run(
            ["sh", run_sh_location],  # Use the absolute path
            shell=True,
            capture_output=True,
            text=True
        )
        # Change back to original working directory
        os.chdir(working_dir)
        output = result.stdout + result.stderr
        return output
        
    else:
        print(f"{current_os} detected - attempting run.sh")
        result = subprocess.run(
            ["sh", run_sh_location],  # Use the absolute path
            shell=True,
            capture_output=True,
            text=True
        )
        # Change back to original working directory
        os.chdir(working_dir)
        output = result.stdout + result.stderr
        return output

#Need to do this to avoid the import from calling main
if __name__ == "__main__":
    main("")