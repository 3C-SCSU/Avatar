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
        [sg.Input(s=25)],
        [sg.Input(s=25)],
        [sg.Text(""), sg.FileBrowse()],
        [sg.Input(s=25, password_char='*')],
    ]

    login_layout = [
        [sg.Col(login_layout_left),
         sg.Push(),
         sg.Col(login_layout_right, element_justification="r")],
    ]

    layout = [
        [sg.Frame("Login", login_layout)],
        [sg.VPush()],
        [sg.Text("Destination Folder:"), sg.Push(), sg.Input(s=25, key="destination_folder")],
        [sg.Text(key='folder_text')],
        [sg.FolderBrowse("Source Folder", target='folder_text'),
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
            pass

    window.close()
