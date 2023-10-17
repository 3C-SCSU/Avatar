import PySimpleGUI as sg

def transfer_data_window():
    layout = [
        [sg.Text("Host:"), sg.Input(s=15)],
        [sg.Text("Username:"), sg.Input(s=15)],
        [sg.Text("Private Key:"), sg.Input(s=15), sg.FileBrowse()],
        [sg.Text("Password:"), sg.Input(s=15, password_char='*')],
    ]

    window = sg.Window("Transfer Data", layout, size=(300, 400), element_justification='c')

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
