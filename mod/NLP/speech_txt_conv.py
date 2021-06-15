import speech_recognition as sr

import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

import gtts
from playsound import playsound

import os 

class recorder: 
    def __init__(self, freq, duration, file_name):
        self.freq = freq
        self.duration = duration
        self.file_name = file_name

    # records user's audio into file
    def record(self):
        self.recording = sd.rec(int(self.duration * self.freq), samplerate=self.freq, channels=2)

        sd.wait()

        # save audio file
        wv.write(self.file_name, self.recording, self.freq, sampwidth=2)

# class converts speech to text
class sph_txt:
    def __init__(self, file_name):
        self.file_name = file_name
    
    def translate(self): 
        r = sr.Recognizer()

        with sr.AudioFile(self.file_name) as f:
            # load audio data
            audio = r.record(f)

            # convert
            self.text = r.recognize_google(audio)

class txt_sph: 
    def __init__(self, text, save_loc): 
        self.text = text 
        self.save_loc = save_loc

    def translate(self): 
        tts = gtts.gTTS(self.text)

        # save audio file
        i = 0
        while True:
            try:
                tts.save(self.save_loc)
                break
            except:
                self.save_loc = self.save_loc[:-4] + str(i) + self.save_loc[-4:]

                i = i + 1

    def play(self): 

        # play file 
        playsound(self.save_loc)
