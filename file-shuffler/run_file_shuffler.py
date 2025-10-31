# Determines the user's current operating system and runs run.sh based on which OS they're on
import os
import platform
import subprocess

os_executed = False


def find_windows_sh_exe_location():
    common_dirs = [
        os.getenv("SystemRoot", "C:\\Windows") + "\\System32",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
    ]

    def search_directory(directory):
        for root, dirs, files in os.walk(directory):
            if "sh.exe" in files:
                return os.path.join(root, "sh.exe")
        return None

    for directory in common_dirs:
        result = search_directory(directory)
        if result:
            return result

    try:
        output = subprocess.check_output("where sh.exe", shell=True, text=True)
        return output.strip().split("\n")[0]  # Return the first found path
    except subprocess.CalledProcessError:
        pass  # Handle the case where `where` command fails

    return None


def convert_to_universal_path(windows_path):
    # Convert backslashes to forward slashes
    universal_path = windows_path.replace("\\", "/")

    # Wrap the path with quotes if it contains spaces
    if " " in universal_path:
        universal_path = f'"{universal_path}"'

    return universal_path


def remove_quotes(input):
    return input.replace('"', "")


def main(path):
    global os_executed
    if not os_executed:
        print("Shuffling: ", path)

        # Get the directory where run_file_shuffler.py is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print("Here is the current file path: ", script_dir)
        # Look for run.sh in the same directory as this script
        run_sh_location = os.path.join(script_dir, "run.sh")

        # Debug print to verify the path
        print(f"Looking for run.sh at: {run_sh_location}")

        if not os.path.exists(run_sh_location):
            print(f"ERROR: run.sh not found at {run_sh_location}")
            return f"ERROR: run.sh not found at {run_sh_location}"

        working_dir = os.getcwd()
        if len(path) > 0:
            universalParameterizedPath = convert_to_universal_path(path)
            universalParameterizedPath = remove_quotes(universalParameterizedPath)
            os.chdir(universalParameterizedPath)
            print(f"Running file shuffler in {universalParameterizedPath}")

        # Get the current operating system
        current_os = platform.system()
        if current_os == "Windows" and not os_executed:
            print("Finding sh.exe to run the script...")
            sh_exe_path = find_windows_sh_exe_location()

            if sh_exe_path and len(sh_exe_path) > 0:
                print(
                    f"\nFound sh.exe at {sh_exe_path}, running the file shuffler... \n"
                )

                print("File shuffler output:")
                sh_exe_path = convert_to_universal_path(sh_exe_path)
            else:
                print(
                    "Failed to find sh.exe path, ensure that sh.exe is installed (check git bash installation)"
                )
                return "Failed to find sh.exe path, ensure that sh.exe is installed (check git bash installation)"

            run_sh_location = convert_to_universal_path(run_sh_location)

            print("Running file shuffler, please wait...\n")

            result = subprocess.run(
                [
                    "powershell",
                    f"& {sh_exe_path} {run_sh_location} {universalParameterizedPath}",
                ],
                shell=True,
                capture_output=True,
                text=True,
            )

            # Change back to original working directory
            os.chdir(working_dir)
            output = result.stdout + result.stderr
            new_lines = [line for line in output.split("\n") if line]

            print("\n".join(new_lines))  # print the output to the console
            os_executed = True

            return "\n".join(new_lines)  # return the output for the gui to consume

        elif current_os == "Linux" and not os_executed:
            process = subprocess.Popen(
                [
                    "sh",
                    run_sh_location,
                    universalParameterizedPath,
                ],  # Use the absolute path
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(run_sh_location),
            )
            # Change back to original working directory
            os.chdir(working_dir)
            stdout, stderr = process.communicate()
            output = stdout + stderr
            os_executed = True
            print("The output is: ", output)
            return output

        elif current_os == "Darwin" and not os_executed:  # macOS
            process = subprocess.Popen(
                [
                    "sh",
                    run_sh_location,
                    universalParameterizedPath,
                ],  # Use the absolute path
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(run_sh_location),
            )
            # Change back to original working directory
            os.chdir(working_dir)
            stdout, stderr = process.communicate()
            output = stdout + stderr
            os_executed = True
            return output

        elif (
            current_os != "Windows"
            and current_os != "linux"
            and current_os != "Darwin"
            and not os_executed
        ):
            print(f"{current_os} detected - attempting run.sh")
            result = subprocess.run(
                [
                    "sh",
                    run_sh_location,
                    universalParameterizedPath,
                ],  # Use the absolute path
                shell=True,
                capture_output=True,
                text=True,
            )
            # Change back to original working directory
            os.chdir(working_dir)
            output = result.stdout + result.stderr
            os_executed = True
            return output


# Need to do this to avoid the import from calling main
if __name__ == "__main__":
    main("")
