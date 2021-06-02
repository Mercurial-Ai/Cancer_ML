import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
import matplotlib.pyplot as plt 
import numpy as np

class gui_audio: 
    def setup(self):

        self.root = ThemedTk(theme='breeze')
        self.root.title("Cancer ML")
        self.root.iconbitmap("cancer_icon.ico")

    def plt_trig(self):
        # (start, stop, step)
        x = np.arange(0, 2*np.pi, 0.1) 
        y = np.sin(x)

        plt.plot(x, y)

        # (start, stop, step)
        x = np.arange(0, 2*np.pi, 0.1)
        y = np.cos(x)
        
        plt.plot(x, y)

    def draw(self): 
        self.root.mainloop()

        canvas = tk.Canvas(self.root)
        canvas.configure(bg="black")
        canvas.pack(fill="both", expand=True)


    def run(self):
        self.setup()
        self.plt_trig()
        self.draw()

ui = gui_audio()
ui.run()