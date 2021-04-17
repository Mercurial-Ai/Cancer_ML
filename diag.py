import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input
from keras import Model

class diagnostic:

    def __init__(self,dataset,target_var):
        self.dataset = dataset
        self.target_var = target_var

    def normalize(self,data,columns):
        min_max_scaler = preprocessing.MinMaxScaler()
        data = min_max_scaler.fit_transform(data)
        df = pd.DataFrame(data, columns=columns)
        return df

    def pre(self):
        if str(type(self.dataset)) == "<class 'str'>":
            df = pd.read_csv(self.dataset)
        else:
            data = self.dataset

            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = pd.Series(data.target)

        # drop rows with missing values
        df = df.dropna()
        x = df

        df = self.normalize(x,df.columns)

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
        self.pre()

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