import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input

class diagnostic:
    def __init__(self,dataset,target_var):
        self.dataset = dataset
        self.target_var = target_var

    def pre(self):
        df = pd.read_csv(self.dataset)

        # drop rows with missing values
        df = df.dropna()
        x = df

        # normalize
        min_max_scaler = preprocessing.MinMaxScaler()
        x = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x,columns=df.columns)

        # feature selection
        corr = df.corr()
        self.features = list(corr.abs().nlargest(10,self.target_var).index)

        df = df[self.features]

        x = df.drop(self.target_var,axis=1)
        y = df[self.target_var]

        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=5)
        x_val,x_test,y_val,y_test = train_test_split(x_test,y_test,test_size=0.5,random_state=5)