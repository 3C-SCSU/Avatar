import tkinter
import customtkinter
import threading

import python

customtkinter.set_appearance_mode(
    "System"
)  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme(
    "dark-blue"
)  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    """
    App class represents the graphical user interface (GUI) for the File Shuffler application

    Attributes:
        folderholder (str): Holds the path of the selected folder
        shuffler_thread (threading.Thread): Thread responsible for shuffling files in the selected folder
        check_shuffler_thread (threading.Thread): Thread responsible for checking the status of the shuffler_thread

    Methods:
        __init__(): Initializes the GUI components and layout
        open_explorer(): Opens a file explorer to allow the user to select a folder
        shuffle(): Initiates the file shuffling process
        hide_widget(): Hides the folder label and shuffle button
        check_shuffling_thread(): Monitors the status of the shuffler_thread and re-enables UI components upon completion
    """

    def __init__(self):
        """
        Initializes the File Shuffler GUI

        - Sets up window configuration, layout, and components
        """

        super().__init__()

        # Placeholders
        self.folderholder = ""
        self.shuffler_thread = None
        self.check_shuffler_thread = None

        # Configure Window
        self.title("File Shuffler")
        self.geometry(f"{834}x{400}")
        self.resizable(0, 0)

        # Configure Grid Layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Sidebar Frame
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=5)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, pady=(1, 1), sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Logo label
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="File Shuffler",
            font=customtkinter.CTkFont(size=20, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Main Window
        self.view = customtkinter.CTkFrame(self, width=100, corner_radius=5)
        self.view.grid(
            row=0, column=1, padx=(20, 20), pady=(20, 20), rowspan=10, sticky="nsew"
        )

        # Select a folder button
        self.select_folder_button = customtkinter.CTkButton(
            self.view, text="Select a directory", command=self.open_explorer
        )
        self.select_folder_button.place(x=250, y=30)

        # Folder frame
        self.folder = customtkinter.CTkLabel(
            self.view,
            width=350,
            height=40,
            fg_color="white",
            text_color="black",
            anchor="w",
        )

        # Remove button
        remove_icon = tkinter.PhotoImage(file="xmark-solid.png").subsample(15, 15)
        self.remove = customtkinter.CTkButton(
            self.folder,
            width=2,
            fg_color="white",
            text="",
            image=remove_icon,
            hover=True,
            compound="right",
            command=self.hide_widget,
        )
        self.remove.grid(row=0, column=0, padx=(370, 20), pady=(10, 10))

        # Shuffle button
        self.shuffle_button = customtkinter.CTkButton(
            self.view, text="shuffle", command=self.shuffle
        )

    def open_explorer(self):
        """
        Opens a file explorer dialog to enable the user to select a folder

        - Updates `folderholder` with the selected folder path
        - Displays the selected folder name on the GUI
        """

        # Get the path of the folder
        self.folderholder = tkinter.filedialog.askdirectory()

        # Clean up the string
        foldername = self.folderholder.split("/")[-1]

        if len(foldername) > 40:
            foldername = foldername[:38] + "..."

        # Display the foldername in the folder frame
        if foldername:
            self.folder.configure(text=" " + foldername)
            self.folder.place(x=30, y=100)
            self.shuffle_button.place(x=450, y=190)

    def void(self):
        """
        Method that does nothing

        - This method serves as a secondary layer of protection to ensure that certain buttons remain disabled
        and do not trigger any unintended actions
        """
        return

    def shuffle(self):
        """
        Initiates the file shuffling process

        - Disables UI components during shuffling
        - Starts the shuffling thread
        - Starts the thread to monitor the shuffling process
        """

        # Disable the button
        self.shuffle_button.configure(state="disable")
        self.shuffle_button.configure(command=self.void)

        self.select_folder_button.configure(state="disable")
        self.select_folder_button.configure(command=self.void)

        self.remove.configure(state="disable")
        self.remove.configure(command=self.void)

        # Initialize the shuffler thread and start it
        self.shuffler_thread = threading.Thread(
            target=python.main, args=(self.folderholder,), daemon=True
        )
        self.shuffler_thread.start()

        # Initialize the thread that monitors the shuffler thread and start it
        self.check_shuffler_thread = threading.Thread(
            target=self.check_shuffling_thread, daemon=True
        )
        self.check_shuffler_thread.start()

    def hide_widget(self):
        """
        Hides the folder label and shuffle button from the GUI

        - Called when the user wants to hide the folder selection
        """

        # Hide the widgets
        self.folder.place_forget()
        self.shuffle_button.place_forget()

    def check_shuffling_thread(self):
        """
        Monitors the status of the file shuffling thread

        - Re-enables UI components upon completion of the shuffling process
        """

        while self.shuffler_thread.is_alive():
            continue

        # Restore the buttons
        self.shuffle_button.configure(state="normal")
        self.shuffle_button.configure(hover=True)
        self.shuffle_button.configure(command=self.shuffle)

        self.select_folder_button.configure(state="normal")
        self.select_folder_button.configure(hover=True)
        self.select_folder_button.configure(command=self.open_explorer)

        self.remove.configure(state="normal")
        self.select_folder_button.configure(hover=True)
        self.remove.configure(command=self.hide_widget)


if __name__ == "__main__":
    app = App()
    app.mainloop()
