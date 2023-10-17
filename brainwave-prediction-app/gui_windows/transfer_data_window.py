import PySimpleGUI as sg

def transfer_data_window():
    layout = [
        [sg.Text("Host:"), sg.Input(s=15)],
        [sg.Text("Username:"), sg.Input(s=15)],
        [sg.Text("Private Key:"), sg.FileBrowse(), sg.Text("")],
        [sg.Text("Password:"), sg.Input(s=15, password_char='*')],
        [sg.HSep()],
        [sg.FileBrowse("Select Files"), sg.Text("")],
        [sg.VPush()],
        [sg.Button("Send"), sg.Button("Cancel")],
    ]

    window = sg.Window("Transfer Data", layout, size=(300, 400), element_justification='c')

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
