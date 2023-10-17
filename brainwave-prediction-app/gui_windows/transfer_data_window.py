import PySimpleGUI as sg

def transfer_data_window():
    layout = [
        [sg.Text("Host:")],
        [sg.Input(s=15)],
    ]

    window = sg.Window("Transfer Data", layout, size=(1200, 800), element_justification='c')

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
