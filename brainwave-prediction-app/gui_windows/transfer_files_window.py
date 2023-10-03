import PySimpleGUI as sg
import os
import subprocess  # To run the sftp.py script

def transfer_files_window():
    # Layout for the transfer files window
    layout = [
        [sg.Text("Select Source Directory:")],
        [sg.InputText(key="-SOURCE-", enable_events=True), sg.FolderBrowse()],
        [sg.Text("Select Target Directory:")],
        [sg.InputText(key="-TARGET-", enable_events=True), sg.FolderBrowse()],
        [sg.Button("Upload"), sg.Button("Cancel")]
    ]

    window = sg.Window("Transfer Files", layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == "Cancel":
            break
        elif event == "Upload":
            source_dir = values["-SOURCE-"]
            target_dir = values["-TARGET-"]

            # Check if both source and target directories are provided
    if source_dir and target_dir:

        # Get the current directory of the script
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Construct the path to sftp.py
        sftp_script_path = os.path.join(script_dir, "..", "file-transfer", "sftp.py")

        # Construct the command to run the sftp.py script with source and target directories as arguments
        cmd = ["python", sftp_script_path, source_dir, target_dir]

        try:
            # Run the sftp.py script with subprocess
            subprocess.run(cmd)
            sg.popup("File Upload Completed!")
        except Exception as e:
            sg.popup_error(f"Error during upload: {str(e)}")
    else:
        sg.popup_error("Please select both source and target directories!")

        window.close()
