import PySimpleGUI as sg
import os

def transfer_files_window():
    # Layout for the file dialog window
    layout = [
        [sg.Text("Select a file:")],
        [sg.InputText(key="-FILENAME-"), sg.FileBrowse(file_types=(("Text Files", "*.txt"), ("All Files", "*.*")))],
        [sg.Button("Open"), sg.Button("Cancel")]
    ]

    window = sg.Window("Open File Dialog", layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == "Cancel":
            break
        elif event == "Open":
            selected_file = values["-FILENAME-"]
            if os.path.isfile(selected_file):
                sg.popup(f"Selected file: {selected_file}")
            else:
                sg.popup_error("Invalid file selected!")

    window.close()