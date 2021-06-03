import tkinter as tk
from tkinter import PhotoImage, ttk
from ttkthemes import ThemedTk

import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image, ImageTk, ImageSequence

import os 

class gui_audio: 
    def setup(self):

        self.root = ThemedTk(theme='breeze')
        self.root.title("Cancer ML")
        self.root.iconbitmap("cancer_icon.ico")

    def draw(self): 

        canvas = tk.Canvas(self.root)
        canvas.configure(bg="white")
        canvas.pack(fill="both", expand=True)

        graph_path = "mod\\GUI\\graphics"

        # get list of graphic imgs avaliable
        graph_imgs = os.listdir(graph_path)

        for img in graph_imgs:

            # only keep files in .png format
            if ".png" not in img: 
                graph_imgs.remove(img)

        i = 0
        for img in graph_imgs: 

            # give entire relative path for every img
            img = os.path.join(graph_path, img)

            graph_imgs[i] = img

            i = i + 1 

        # start with first photo
        tk_photo = tk.PhotoImage(file=graph_imgs[0])
        canvas.create_image(20, 20, anchor=tk.NW, image=tk_photo)

        while 'normal' == self.root.state():
            for img in graph_imgs[1:]: 
                self.root.update()
                self.root.after(300, tk_photo.config(file=img))

        self.root.mainloop()

    def run(self):
        self.setup()
        self.draw()

ui = gui_audio()
ui.run()