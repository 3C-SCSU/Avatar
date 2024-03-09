__Program__     = "ShufflerGUI"    
__programers__   = "Madison Arndt, Stephen Wagner, Matthew Mokoro"
__Date__        = "2/22/2024"

"""
this gui can be used independently by invoking main()
it also can be used as a frame in a main gui
"""


_textFG = "white"
_windowBG = "#64778d"
_btnBG = "#283b5b"

#"""
try:
    #import if imported from another file
    from . import python_source_code as shuffler
except:
    #import from this file
    import python_source_code as shuffler
#"""

#import tkinter files used
import tkinter  as tk
from tkinter import messagebox
from tkinter import filedialog as fd

#icon used in frames
import os
_FP = os.path.dirname(__file__)
icon = f"{_FP}/icon.ico"

class ShufflerGUI(tk.Frame):
    filePath = None
    def __init__(self,parent,fbg = _windowBG,bbg=_btnBG,fg=_textFG,
                 *args,**kw):
        tk.Frame.__init__(self,parent,bg=fbg,*args,**kw)
        #path lbl
        self.filePathlbl = tk.Label(self,bg = fbg,fg = fg,
                                   font = (f'timesnewroman 9 bold'),
                                   text = f"file path: {self.filePath}")
        self.filePathlbl.pack(fill="x")
        
        frm = tk.Frame(self,bg=fbg,*args,**kw)
        frm.pack(fill="both",expand=True)
        #file path btn
        self.openfilePathbtn = tk.Button(frm,bg = bbg,fg = fg,
                                         text = "select file path",
                     font = (f'timesnewroman 9 bold'),
                     command=self.openFilePath)
        self.openfilePathbtn.pack(side="left")
        #run btn
        self.runbtn = tk.Button(frm,bg = bbg,fg = fg,
                                text = "Run Shuffler",command=self.run,
                                font = (f'timesnewroman 9 bold'))
        self.runbtn.pack(side="right")

    def openFilePath(self): #Stephen
        self.filePath = fd.askdirectory()

        if self.filePath:
            self.filePathlbl.config(text=f"File path: {self.filePath}")

        else:
            messagebox.showwarning("Warning", "No folder was selected. \
Please select a folder.")
            self.filePath=None

    def run(self): #Matthew
        if self.filePath:
            try:
                shuffler.main(self.filePath)
            except Exception as e:
                messagebox.showwarning('Warning', f'Error: {e}')
        else:
            messagebox.showwarning("Warning", "No folder was selected. \
Please select a folder.")
        

    def shuffle(self):
        self.Run()


def main():
    root = tk.Tk()
    root.title("File Shuffler")#title of widow
    root.geometry(f"300x400")
    root.iconbitmap(default = icon)

    frm = ShufflerGUI(root)
    frm.pack(fill="both",expand=True)

    #run GUI
    root.mainloop()
        


if __name__ == "__main__":
    main()
















