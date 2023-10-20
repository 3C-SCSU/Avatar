import PySimpleGUI as sg
from .sftp import fileTransfer

def transfer_data_window():
    login_layout_left = [
        [sg.Text("Host:")],
        [sg.Text("Username:")],
        [sg.Text("Private Key:")],
        [sg.Text("Password:")],
    ]

    login_layout_right = [
        [sg.Input(s=25, key="host")],
        [sg.Input(s=25, key="username")],
        [sg.FileBrowse(key="private_key")],
        [sg.Input(s=25, password_char='*', key="private_key_pass")],
    ]

    login_layout = [
        [sg.Col(login_layout_left),
         sg.Push(),
         sg.Col(login_layout_right, element_justification="r")],
    ]

    layout = [
        [sg.Frame("Login", login_layout)],
        [sg.VPush()],
        [sg.Text("Source Folder:"), sg.Push(), sg.Text(key='source_folder')],
        [sg.Text("Destination Folder:"), sg.Push(), sg.Input(s=25, key="destination_folder")],
        [sg.FolderBrowse("Source Folder", target='source_folder'),
         sg.Push(),
         sg.Button("Send"),
         sg.Button("Cancel")],
    ]

    window = sg.Window("Transfer Data", layout, size=(350, 250), element_justification='c')

    while True:
        event, values = window.read()

        if event in [sg.WIN_CLOSED, "Cancel"]:
            break
        if event == "Send":
            for value in ["host", "username", "private_key", "private_key_pass", "Source Folder", "destination_folder"]:
                if values.get(value) in ["", None]:
                    break
            else:
                try:
                    conn = fileTransfer(
                        values["host"],
                        values["username"],
                        values["private_key"],
                        values["private_key_pass"])
                    conn.transfer(values["Source Folder"], values["destination_folder"])
                    sg.popup("File transfer complete")
                except Exception as err:
                    sg.popup_error("Failed to transfer files")

    window.close()
