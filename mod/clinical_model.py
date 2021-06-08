from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers import Dense
import keras
import pandas as pd 
from mod.percentage_accuracy import percentageAccuracy

class clinical:
    def __init__(self, data_file, mainPath, target_vars, load_fit, save_fit, save_location, epochs_num, activation_function):
        self.data_file = data_file
        self.mainPath = mainPath
        self.target_vars = target_vars
        self.load_fit = load_fit
        self.save_fit = save_fit
        self.save_location = save_location
        self.epochs_num = epochs_num
        self.activation_function = activation_function

    def checkBinary(self, target_var):  

        orgPD = pd.read_csv(self.mainPath)
        orgPD = orgPD.dropna()

        # check if param is a list of multiple vars 
        if str(type(target_var)) == "<class 'list'>" and len(target_var) > 1:

            for vars in target_var: 

                # initialize list to hold bools 
                areBinary = []
            
                col = list(orgPD[vars])

                # remove duplicates 
                col = list(set(col))

                # check if data is numerical 
                for vals in col: 
                    if str(type(vals)) == "<class 'int'>" or str(type(vals)) == "<class 'float'>": 
                        numeric = True
                    else: 
                        numeric = False 

                if not numeric: 

                    if len(col) == 2: 
                        isBinary = True
                    else: 
                        isBinary = False 

                    areBinary.append(isBinary)
                else: 
                    areBinary = False

            isBinary = areBinary 

        else: 

            col = list(orgPD[target_var])

            # remove duplicates 
            col = list(set(col))

            # check if original data is numerical
            for vals in col: 
                if str(type(vals)) == "<class 'int'>" or str(type(vals)) == "<class 'float'>": 
                    numeric = True
                else: 
                    numeric = False 
            
            if not numeric: 
                if len(col) == 2: 
                    isBinary = True
                else: 
                    isBinary = False 

            else: 
                isBinary = False

        return isBinary

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

        self.isBinary = self.checkBinary(self.target_vars)

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

        if self.isBinary: 
            y_list = list(y)

            # remove duplicates to identify binary vals
            y_list = list(set(y_list))

            # sort vals in ascending order
            y_list.sort()

            binary_dict = {y_list[0]: 0, y_list[1]: 1}

            i = 0
            new_y = pd.Series([])
            for val in y: 
                conv_y = binary_dict[val]

                new_y[i] = conv_y
                i = i + 1 

            y = new_y

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

    def post(self):

        # utilize validation data
        prediction = self.model.predict(self.X_val,batch_size=1)

        roundedPred = np.around(prediction, 0)

        if self.multiple_targets == False and roundedPred.ndim == 1:
            i = 0
            for vals in roundedPred:
                if int(vals) == -0:
                    vals = abs(vals)
                    roundedPred[i] = vals

                i = i + 1

        else:
            preShape = roundedPred.shape

            # if array has multiple dimensions, flatten the array
            roundedPred = roundedPred.flatten()

            i = 0
            for vals in roundedPred:
                if int(vals) == -0:
                    vals = abs(vals)
                    roundedPred[i] = vals

                i = i + 1

            if len(preShape) == 3:
                if preShape[2] == 1:
                    # reshape array to previous shape without the additional dimension
                    roundedPred = np.reshape(roundedPred, preShape[:2])
                else:
                    roundedPred = np.reshape(roundedPred, preShape)

            else:
                roundedPred = np.reshape(roundedPred, preShape)

            print("Validation Metrics")
            print("- - - - - - - - - - - - - Unrounded Prediction - - - - - - - - - - - - -")
            print(prediction)
            print("- - - - - - - - - - - - - Rounded Prediction - - - - - - - - - - - - -")
            print(roundedPred)
            print("- - - - - - - - - - - - - y val - - - - - - - - - - - - -")
            print(self.y_val)

            if str(type(prediction)) == "<class 'list'>":
                prediction = np.array([prediction])

            percentAcc = percentageAccuracy(roundedPred, self.y_val)

            print("- - - - - - - - - - - - - Percentage Accuracy - - - - - - - - - - - - -")
            print(percentAcc)

            self.resultList = []

            self.resultList.append(str(prediction))
            self.resultList.append(str(roundedPred))
            self.resultList.append(str(self.y_val))
            self.resultList.append(str(percentAcc))

            # utilize test data
            prediction = self.model.predict(self.X_test, batch_size=1)

            roundedPred = np.around(prediction, 0)

            if self.multiple_targets == False and roundedPred.ndim == 1:
                i = 0
                for vals in roundedPred:
                    if int(vals) == -0:
                        vals = abs(vals)
                        roundedPred[i] = vals

                    i = i + 1
            else:
                preShape = roundedPred.shape

                # if array has multiple dimensions, flatten the array
                roundedPred = roundedPred.flatten()

                i = 0
                for vals in roundedPred:
                    if int(vals) == -0:
                        vals = abs(vals)
                        roundedPred[i] = vals

                    i = i + 1

                if len(preShape) == 3:
                    if preShape[2] == 1:
                        # reshape array to previous shape without the additional dimension
                        roundedPred = np.reshape(roundedPred, preShape[:2])
                    else:
                        roundedPred = np.reshape(roundedPred, preShape)
                else:
                    roundedPred = np.reshape(roundedPred, preShape)

            print("Test Metrics")
            print("- - - - - - - - - - - - - Unrounded Prediction - - - - - - - - - - - - -")
            print(prediction)
            print("- - - - - - - - - - - - - Rounded Prediction - - - - - - - - - - - - -")
            print(roundedPred)
            print("- - - - - - - - - - - - - y test - - - - - - - - - - - - -")
            print(self.y_test)

            if str(type(prediction)) == "<class 'list'>":
                prediction = np.array([prediction])

            percentAcc = percentageAccuracy(roundedPred, self.y_test)

            print("- - - - - - - - - - - - - Percentage Accuracy - - - - - - - - - - - - -")
            print(percentAcc)

            self.resultList.append(str(prediction))
            self.resultList.append(str(roundedPred))
            self.resultList.append(str(self.y_test))
            self.resultList.append(str(percentAcc))

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

                self.model = keras.Model(inputs=input, outputs=output)

                self.model.compile(optimizer='SGD',
                              loss='mean_absolute_error',
                              metrics=['accuracy'])

                fit = self.model.fit(self.X_train, self.y_train, epochs=self.epochs_num, batch_size=5)

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
                self.model = keras.Model(input, output)

                self.model.compile(optimizer='SGD',
                              loss='mean_squared_error',
                              metrics=['accuracy'])

                fit = self.model.fit(self.X_train, self.y_train, epochs=self.epochs_num, batch_size=32)

                if self.save_fit == True:
                    self.model.save(self.save_location)
        else:
            self.model = keras.models.load_model(self.save_location)

        self.post()
