import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import json

from speech_txt_conv import recorder, sph_txt, txt_sph

class NLP: 

    def get_message(self): 
        r = recorder(freq=44100, duration=7, file_name="user_recording.wav")
        r.record()

        conv = sph_txt(r.file_name)
        conv.translate()
        self.text = conv.text

    # function to partition text into sentences, words, etc 
    def partition(self): 

        # separate sentences 
        self.text = sent_tokenize(self.text)

        # get list of words for each sentence
        word_sent_list = []
        for sent in self.text: 

            # get list of words 
            self.text = word_tokenize(sent)

            word_sent_list.append(self.text)

        self.text = word_sent_list

    # function that filters out stopwords (an, in, is, etc.)
    def filterStops(self):

        stops = set(stopwords.words("english"))
        self.filtered_list = []
        for sent in self.text: 
            for word in sent: 
                if word.casefold() not in stops: 
                    self.filtered_list.append(word)

    # function to extract primitive meaning of words
    def stem(self):
        stemmer = WordNetLemmatizer()
        
        self.stemmed_words = []
        for word in self.filtered_list: 
            stem_word = stemmer.lemmatize(word)
            self.stemmed_words.append(stem_word)

    # function to classify different parts of the text
    def tag(self):
        self.tags = nltk.pos_tag(self.stemmed_words)

        # convert list of tuples to dict
        i = 0 
        tag_dict = {}
        for tup in self.tags: 
            word = tup[0]
            tag = tup[1]

            # check if tag already in dict
            if tag in tag_dict: 
                tag = tag + str(i)
            
            tag_dict[tag] = word

            i = i + 1 

        self.tags = tag_dict
        print(self.tags)

    def remove_breaks(self, val_list):
        i = 0
        for var in val_list: 
            new_var = var.replace('\n','')
            val_list[i] = new_var
            i = i + 1 
        
        return val_list

    def get_info(self): 
        # identify variable, value, and unit

        NN_keys = []
        for key in self.tags.keys(): 
            if key[:2] == 'NN': 
                NN_keys.append(key)

        # read list of possible variables 
        f = open("mod\\NLP\\data\\variables.txt")
        var_list = f.readlines()

        # remove line breakers throughout var list
        var_list = self.remove_breaks(var_list)
        
        # check if a variable name is inside of text
        for var in var_list: 
            if var in list(self.tags.values()): 
                self.variable = var
                break
            else: 
                self.variable = "variable not found"

        # identify numerical values in text for var value
        for var in self.tags.values(): 
            if var.isnumeric(): 
                self.value = var

        # identify unit by checking in units.txt
        f = open("mod\\NLP\\data\\units.txt")
        unit_list = f.readlines()

        unit_list = self.remove_breaks(unit_list)

        for unit in unit_list: 
            if unit in self.tags.values():
                self.val_unit = unit

    def respond(self):
        s = txt_sph(str(self.variable), "bot_recording.mp3")
        s.translate()
        s.play()

    def run(self): 
        self.get_message()
        self.partition()
        self.filterStops()
        self.stem()
        self.tag()
        self.get_info()
        self.respond()

s = NLP()
s.run()
    