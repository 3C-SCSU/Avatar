import PySimpleGUI as sg
import os
import configparser
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." "file-transfer"))
from sftp import fileTransfer

config = configparser.ConfigParser() # Used for saving and loading login data for the target
config.optionxform = str # Make the saved keys case-sensitive

class TransferData:
    def transfer_files_window(self):
        # Layout for the transfer files window
        column_to_be_centered = [
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
            [sg.Button("Save Config"), sg.Button("Load Config"), sg.Button("Clear Config"), sg.Button("Upload")]
        ]
        layout = [[sg.VPush()],
          [sg.Push(), sg.Column(column_to_be_centered), sg.Push()],
          [sg.VPush()]]
        
        tab = sg.Tab('Transfer Data', layout, key='Transfer Data')
        return tab
    
    def buttonLoop (self, window1, event, values):
        if event == "Upload":
            try:
                # Attempt to open a server connection
                svrcon = fileTransfer(values["-HOST-"], values["-USERNAME-"], values["-PRIVATE_KEY-"], values["-PRIVATE_KEY_PASS-"], values["-IGNORE_HOST_KEY-"])
                source_dir = values["-SOURCE-"]
                target_dir = values["-TARGET-"]

                # Check if both source and target directories are provided
                if source_dir and target_dir:
                    try:
                        # Attempt to transfer the files
                        svrcon.transfer(str(source_dir), (target_dir))
                        sg.popup("File Upload Completed!")
                    except Exception as e:
                        sg.popup_error(f"Error during upload: {str(e)}")
                else:
                    sg.popup_error("Please ensure that all fields have been filled!")
            except Exception as e:
                sg.popup_error(f"Error during upload (check your login info): {str(e)}")

        elif event == "Save Config":
            # Manually open a file dialog
            selected_file = sg.popup_get_file(message="Save config file", save_as=True, no_window=True, default_extension="ini", file_types=(("ini", ".ini"),))

            # The login data that will be saved
            config['data'] = {
                "-HOST-": values["-HOST-"],
                "-USERNAME-": values["-USERNAME-"],
                "-PRIVATE_KEY-": values["-PRIVATE_KEY-"],
                "-IGNORE_HOST_KEY-": values["-IGNORE_HOST_KEY-"],
                "-SOURCE-": values["-SOURCE-"],
                "-TARGET-": values["-TARGET-"],
            }          

            # Write the data to disk at the selected location
            if selected_file:
                with open(selected_file, 'w') as configfile:
                    config.write(configfile)

        elif event == "Load Config":
            # Manually open a file dialog
            selected_file = sg.popup_get_file(message="Save config file", no_window=True, file_types=(("ini", ".ini"),))
            
            # The original login data
            oldData = {
                "-HOST-": values["-HOST-"],
                "-USERNAME-": values["-USERNAME-"],
                "-PRIVATE_KEY-": values["-PRIVATE_KEY-"],
                "-IGNORE_HOST_KEY-": values["-IGNORE_HOST_KEY-"],
                "-SOURCE-": values["-SOURCE-"],
                "-TARGET-": values["-TARGET-"],
            }   

            try:
                if selected_file:
                    # Attempt to read the selected file
                    config.read(selected_file)
                    # Use the loaded data to set the values
                    for key, value in config['data'].items():
                        window1[key].update(value=value)

            except Exception as e:
                # Reset the values back to the original values
                for key, value in oldData.items():
                    window1[key].update(value=value)
                sg.popup_error(f"Config file error: {str(e)}")

        elif event == "Clear Config":
            window1["-HOST-"].update(value="")
            window1["-USERNAME-"].update(value="")
            window1["-PRIVATE_KEY-"].update(value="")
            window1["-PRIVATE_KEY_PASS-"].update(value="")
            window1["-IGNORE_HOST_KEY-"].update(value=True)
            window1["-SOURCE-"].update(value="")
            window1["-TARGET-"].update(value="")

        return
