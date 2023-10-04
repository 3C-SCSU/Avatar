import PySimpleGUI as sg
import os
import configparser
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "file-transfer"))
from sftp import fileTransfer

config = configparser.ConfigParser()
config.optionxform = str

def transfer_files_window(window1):
    # Layout for the transfer files window
    layout = [
        [sg.Text("Target IP:")],
        [sg.InputText(key="-HOST-", enable_events=True)],
        [sg.Text("Target Username")],
        [sg.InputText(key="-USERNAME-", enable_events=True)],
        [sg.Text("Target Password")],
        [sg.InputText(key="-PRIVATE_KEY_PASS-", enable_events=True, password_char='*')],
        [sg.Text("Private Key Directory:")],
        [sg.InputText(key="-PRIVATE_KEY-", enable_events=True), sg.FileBrowse()],
        [sg.Checkbox("Ignore Host Key", key="-IGNORE_HOST_KEY-", default=True)],
        [sg.Text("Source Directory:")],
        [sg.InputText(key="-SOURCE-", enable_events=True), sg.FolderBrowse()],
        [sg.Text("Target Directory:")],
        [sg.InputText(key="-TARGET-", enable_events=True, default_text="/home/")],
        [sg.Button("Save Config"), sg.Button("Load Config"), sg.Button("Upload"), sg.Button("Cancel")]
    ]

    transfer_files_window = sg.Window("Transfer Files", layout)

    while True:
        event, values = transfer_files_window.read()

        if event in (sg.WIN_CLOSED, 'Quit') or event == "Cancel":
            break
        elif event == "Upload":
            svrcon = None
            source_dir = None
            target_dir = None
            
            try:
                svrcon = fileTransfer(values["-HOST-"], values["-USERNAME-"], values["-PRIVATE_KEY-"], values["-PRIVATE_KEY_PASS-"], values["-IGNORE_HOST_KEY-"])
                source_dir = values["-SOURCE-"]
                target_dir = values["-TARGET-"]

                # Check if both source and target directories are provided
                if source_dir and target_dir:
                    try:
                        # Run the sftp.py script
                        svrcon.transfer(str(source_dir), (target_dir))
                        sg.popup("File Upload Completed!")
                    except Exception as e:
                        sg.popup_error(f"Error during upload: {str(e)}")
                else:
                    sg.popup_error("Please select both source and target directories!")
            except Exception as e:
                sg.popup_error(f"Error during upload (check your login info): {str(e)}")

        elif event == "Save Config":
            selected_file = sg.popup_get_file(message="Save config file", save_as=True, no_window=True, default_extension="ini", file_types=(("ini", ".ini"),))
            config['data'] = {
                "-HOST-": values["-HOST-"],
                "-USERNAME-": values["-USERNAME-"],
                "-PRIVATE_KEY-": values["-PRIVATE_KEY-"],
                "-IGNORE_HOST_KEY-": values["-IGNORE_HOST_KEY-"],
                "-SOURCE-": values["-SOURCE-"],
                "-TARGET-": values["-TARGET-"],
            }          

            with open(selected_file, 'w') as configfile:
                config.write(configfile)

        elif event == "Load Config":
            selected_file = sg.popup_get_file(message="Save config file", no_window=True, file_types=(("ini", ".ini"),))
            
            oldData = {
                "-HOST-": values["-HOST-"],
                "-USERNAME-": values["-USERNAME-"],
                "-PRIVATE_KEY-": values["-PRIVATE_KEY-"],
                "-IGNORE_HOST_KEY-": values["-IGNORE_HOST_KEY-"],
                "-SOURCE-": values["-SOURCE-"],
                "-TARGET-": values["-TARGET-"],
            }   

            try:
                config.read(selected_file)
                for key, value in config['data'].items():
                    transfer_files_window[key].update(value=value)

            except Exception as e:
                sg.popup_error(f"Config file error: {str(e)}")
                for key, value in oldData.items():
                    transfer_files_window[key].update(value=value)

    window1.un_hide()
    transfer_files_window.close() 
