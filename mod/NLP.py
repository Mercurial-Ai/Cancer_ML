import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
nltk.download("punkt")
nltk.download("stopwords")

class NLP: 
    def __init__(self, text):
        self.text = text

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

    # function to classify different parts of the text
    def tag(self):
        self.tags = nltk.pos_tag(self.filtered_list)

    # function to extract primitive meaning of words
    def stem(self):
        stemmer = WordNetLemmatizer()
        
        self.stemmed_words = []
        for word in self.filtered_list: 
            stem_word = stemmer.lemmatize(word)
            self.stemmed_words.append(stem_word)

    def run(self): 
        self.partition()
        self.filterStops()
        self.tag()
        self.stem()
    