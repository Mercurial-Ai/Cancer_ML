from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers import Dense
import keras

class clinical:
    def __init__(self, data_file, target_vars, load_fit, save_fit, save_location, epochs_num, activation_function):
        self.data_file = data_file
        self.target_vars = target_vars
        self.load_fit = load_fit
        self.save_fit = save_fit
        self.save_location = save_location
        self.epochs_num = epochs_num
        self.activation_function = activation_function

    def feature_selection(self, dataset, target_vars, num_features):
        if self.multiple_targets == False:
            features = list(dataset.corr().abs().nlargest(num_features, target_vars).index)
        else:
            features = []
            for vars in target_vars:
                feature = list(dataset.corr().abs().nlargest(num_features, vars).values[:, dataset.shape[1]-1])
                features.append(feature)

            features = sum(features, [])

        return features

    def hasNan(self,array):
        # function checks for nans/non-compatible objects

        nan = np.isnan(array)
        for arr in nan:
            if array.ndim == 2:
                for bool in arr:
                    if bool:
                        containsNan = True
                    else:
                        containsNan = False
            elif array.ndim == 1:
                if arr:
                    containsNan = True
                else:
                    containsNan = False

        # check that all data is floats or integers
        if array.ndim == 1:
            typeList = []
            for vals in array:
                valType = str(type(vals))
                typeList.append(valType)

            for types in typeList:
                if types != "<class 'numpy.float64'>" and types != "<class 'numpy.int64'>":
                    containsNan = True

        if containsNan:
            print("Data contains nan values")
        else:
            print("Data does not contain nan values")

    def pre(self):

        # initialize bool as false
        self.multiple_targets = False

        if str(type(self.target_vars)) == "<class 'list'>" and len(self.target_vars) > 1:
            self.multiple_targets = True

        if self.multiple_targets == False:
            # retrieve top 10 most correlated features to utilize
            features = list(self.feature_selection(self.data_file, self.target_vars, 10))
        else:
            # initialize list
            features = []

            # make list with top 10 most correlated features from both vars.
            # Ex. 20 total features for 2 target vars
            for vars in self.target_vars:
                featuresVar = list(self.feature_selection(self.data_file,vars,10))
                features = features + featuresVar

            # remove duplicates
            features = list(set(features))

        # only use features determined by feature_selection
        self.data_file = self.data_file[self.data_file.columns.intersection(features)]

        df = self.data_file

        # x data
        X = df.drop(self.target_vars, axis=1)

        # y data
        y = df.loc[:, self.target_vars]

        # partition data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        # partition val/test
        self.X_test, self.X_val = train_test_split(self.X_test, test_size=0.5, random_state=34)
        self.y_test, self.y_val = train_test_split(self.y_test, test_size=0.5, random_state=34)

        # normalize data
        min_max_scaler = MinMaxScaler()
        self.X_train = min_max_scaler.fit_transform(self.X_train)
        self.X_test = min_max_scaler.fit_transform(self.X_test)
        self.X_val = min_max_scaler.fit_transform(self.X_val)

        if self.multiple_targets:
            self.y_test = min_max_scaler.fit_transform(self.y_test)
            self.y_train = min_max_scaler.fit_transform(self.y_train)
            self.y_val = min_max_scaler.fit_transform(self.y_val)

        if str(type(self.y_train)) == "<class 'pandas.core.frame.DataFrame'>":
            self.y_train = self.y_train.to_numpy()

        if str(type(self.y_test)) == "<class 'pandas.core.frame.DataFrame'>":
            self.y_test = self.y_test.to_numpy()

        # check y_train for NANs
        self.hasNan(self.y_train)

    def NN(self):
        self.pre()

        if not self.load_fit:
            if str(type(self.target_vars))=="<class 'list'>" and len(self.target_vars) > 1:
                input = keras.Input(shape=self.X_train.shape[1],)

                x = Dense(10, activation=self.activation_function)(input)
                x = Dense(10, activation=self.activation_function)(x)
                x = Dense(6, activation=self.activation_function)(x)
                x = Dense(4, activation=self.activation_function)(x)
                x = Dense(4, activation=self.activation_function)(x)
                output = Dense(len(self.target_vars), activation=self.activation_function)(x)

                model = keras.Model(inputs=input, outputs=output)

                model.compile(optimizer='SGD',
                              loss='mean_absolute_error',
                              metrics=['accuracy'])

                fit = model.fit(self.X_train, self.y_train, epochs=self.epochs_num, batch_size=5)

            else:
                print(self.X_train.shape[1])

                # set input shape to dimension of data
                input = keras.layers.Input(shape=(self.X_train.shape[1],))

                x = Dense(9, activation=self.activation_function)(input)
                x = Dense(9, activation=self.activation_function)(x)
                x = Dense(6, activation=self.activation_function)(x)
                x = Dense(4, activation=self.activation_function)(x)
                x = Dense(2, activation=self.activation_function)(x)
                output = Dense(1, activation='linear')(x)
                model = keras.Model(input, output)

                model.compile(optimizer='SGD',
                              loss='mean_squared_error',
                              metrics=['accuracy'])

                fit = model.fit(self.X_train, self.y_train, epochs=self.epochs_num, batch_size=32)

                if self.save_fit == True:
                    model.save(self.save_location)
        else:
            model = keras.models.load_model(self.save_location)
