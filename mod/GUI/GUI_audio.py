import tkinter as tk
from tkinter import PhotoImage, ttk
from ttkthemes import ThemedTk

import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image, ImageTk, ImageSequence

import os 
import sys
sys.path.append('mod//NLP')

if __name__ == "__main__": 
    # do not run when imported
    from speech_txt_conv import sph_txt, txt_sph, recorder

class gui_audio: 
    def setup(self):

        self.root = ThemedTk(theme='breeze')
        self.root.title("Cancer ML")
        self.root.iconbitmap("cancer_icon.ico")

    def start_cmd(self): 

        record = recorder(44100, 5, 'user_recording.wav')
        record.record()

        translator = sph_txt('user_recording.wav')
        translator.translate()
        self.text = translator.text
        self.respond()

    def respond(self): 
        conv = txt_sph(self.text, 'bot_recording.mp3')
        conv.translate()
        conv.play()

    def draw(self): 

        self.start = tk.Button(text="Start", command=self.start_cmd)
        self.start.place(height=80, width=100, relx=0.5, rely=0.5)

        self.root.mainloop()

    def run(self):
        self.setup()
        self.draw()

ui = gui_audio()
ui.run()
