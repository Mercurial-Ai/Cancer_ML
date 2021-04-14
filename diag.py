import pandas as pd
from sklearn import preprocessing

class diagnostic:
    def __init__(self,dataset):
        self.dataset = dataset

    def pre(self):
        df = pd.read_csv(self.dataset)

        # drop rows with missing values
        df = df.dropna()

        # normalize
        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(df)
