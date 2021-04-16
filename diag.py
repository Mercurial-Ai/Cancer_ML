import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input
from keras import Model

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

        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size=0.4,random_state=5)
        self.x_val,self.x_test,self.y_val,self.y_test = train_test_split(self.x_test,self.y_test,test_size=0.5,random_state=5)

    # function for val/test pred
    def post(self,dataX,dataY,model):
        results = model.evaluate(dataX, dataY, batch_size=32)
        pred = model.predict(dataX)

    def model(self):
        input = Input(shape=self.x_train.shape[1],)
        x = Dense(9,activation="relu")(input)
        x = Dense(6,activation="relu")(x)
        x = Dense(3,activation="relu")(x)
        output = Dense(1,activation="linear")(x)

        model = Model(input,output)

        model.compile(optimizer="SGD",
                      loss="mean_squared_error",
                      metrics=["accuracy"])

        fit = model.fit(self.x_train,self.y_train,epochs=20,batch_size=32)

        self.post(self.x_val, self.y_val, model)
        self.post(self.x_test,self.y_test,model)


