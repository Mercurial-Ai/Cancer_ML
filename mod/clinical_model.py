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

    def percentageAccuracy(self, iterable1, iterable2):

        def roundList(iterable):

            if str(type(iterable)) == "<class 'tensorflow.python.framework.ops.EagerTensor'>":
                iterable = iterable.numpy()
            roundVals = []
            if int(iterable.ndim) == 1:
                for i in iterable:
                    i = round(i, 0)
                    roundVals.append(i)

            elif int(iterable.ndim) == 2:
                for arr in iterable:
                    for i in arr:
                        i = round(i, 0)
                        roundVals.append(i)

            elif int(iterable.ndim) == 3:
                for dim in iterable:
                    for arr in dim:
                        for i in arr:
                            i = round(i, 0)
                            roundVals.append(i)

            elif int(iterable.ndim) == 4:
                for d in iterable:
                    for dim in d:
                        for arr in dim:
                            for i in arr:
                                i = round(i, 0)
                                roundVals.append(i)

            else:
                print("Too many dimensions--ERROR")

            return roundVals

        rounded1 = roundList(iterable1)
        rounded2 = roundList(iterable2)

        # remove negative zeros from lists
        i = 0
        for vals in rounded1:
            if int(vals) == -0 or int(vals) == 0:
                vals = abs(vals)
                rounded1[i] = vals

            i = i + 1

        i = 0
        for vals in rounded2:
            if int(vals) == -0 or int(vals) == 0:
                vals = abs(vals)
                rounded2[i] = vals

            i = i + 1

        numCorrect = len([i for i, j in zip(rounded1, rounded2) if i == j])

        listLen = len(rounded1)

        percentCorr = numCorrect / listLen
        percentCorr = percentCorr * 100

        percentCorr = round(percentCorr, 2)

        return percentCorr

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

            percentAcc = self.percentageAccuracy(roundedPred, self.y_val)

            print("- - - - - - - - - - - - - Percentage Accuracy - - - - - - - - - - - - -")
            print(percentAcc)

            resultList.append(str(prediction))
            resultList.append(str(roundedPred))
            resultList.append(str(y_val))
            resultList.append(str(percentAcc))

            # utilize test data
            prediction = model.predict(X_test, batch_size=1)

            roundedPred = np.around(prediction, 0)

            if multiple_targets == False and roundedPred.ndim == 1:
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
            print(y_test)

            if str(type(prediction)) == "<class 'list'>":
                prediction = np.array([prediction])

            percentAcc = percentageAccuracy(roundedPred, y_test)

            print("- - - - - - - - - - - - - Percentage Accuracy - - - - - - - - - - - - -")
            print(percentAcc)

            resultList.append(str(prediction))
            resultList.append(str(roundedPred))
            resultList.append(str(y_test))
            resultList.append(str(percentAcc))

            if multiple_targets == True and str(type(isBinary)) == "<class 'list'>":

                # initialize var as error message
                decodedPrediction = "One or all of the target variables are non-binary and/or numeric"

                i = 0
                for bools in isBinary:
                    if bools == True:
                        decodedPrediction = decode(prediction[0, i], targetDict)
                    i = i + 1
            else:
                if isBinary:
                    decodedPrediction = decode(prediction, targetDict)
                else:
                    decodedPrediction = "One or all of the target variables are non-binary and/or numeric"

            print("- - - - - - - - - - - - - Translated Prediction - - - - - - - - - - - - -")
            print(decodedPrediction)

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
