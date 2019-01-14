from CMD.Single_Run import *
from tkinter import *
from PIL import ImageTk as itk


class SOMView:
    def __init__(self, som):
        self.window = Toplevel()
        self.window.title("Self-Organised Map display")
        self.canvas = Canvas(self.window, width=300, height=300, bg="ivory")
        self.canvas.grid(row=0, column=0, columnspan=10, rowspan=10, padx=10, pady=10)

        self.SOM = som

        self.draw_SOM()

        self.canvas.update()
        self.window.mainloop()

    def draw_SOM(self):
        self.im = display_som(self.SOM.get_som_as_list())
        self.im = self.im.resize((300, 300))
        self.ph = itk.PhotoImage(self.im)
        self.canvas.create_image(0, 0, image=self.ph, anchor=NW)

    def refresh(self):
        self.canvas.delete("all")
        self.draw_SOM()
        self.canvas.update()
