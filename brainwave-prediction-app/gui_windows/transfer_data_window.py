import PySimpleGUI as sg

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
        [sg.Text(key='folder')],
        [sg.FolderBrowse(target='folder'),
         sg.Push(),
         sg.Button("Send"),
         sg.Button("Cancel")],
    ]

    window = sg.Window("Transfer Data", layout, size=(350, 200), element_justification='c')

    while True:
        event, values = window.read()

        if event in [sg.WIN_CLOSED, "Cancel"]:
            break

    window.close()
