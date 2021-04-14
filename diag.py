import pandas as pd

class diagnostic:
    def __init__(self,dataset):
        self.dataset = dataset

    def pre(self):
        df = pd.read_csv(self.dataset)

        print(df.head())
