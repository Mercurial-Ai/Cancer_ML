from numpy.core.numeric import True_
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import keras
from keras import layers
import os 
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import psutil
import matplotlib.pyplot as plt

class image_model: 
    def __init__(self, model_save_loc, data_file, target_vars, epochs_num, load_numpy_img, img_array_save, load_fit, img_dimensions, img_id_name_loc, ID_dataset_col, useCNN, data_save_loc, save_figs, show_figs): 
        self.model_save_loc = model_save_loc
        self.data_file = data_file 
        self.target_vars = target_vars
        self.epochs_num = epochs_num
        self.load_numpy_img = load_numpy_img
        self.img_array_save = img_array_save
        self.load_fit = load_fit
        self.img_dimensions = img_dimensions
        self.img_id_name_loc = img_id_name_loc
        self.ID_dataset_col = ID_dataset_col
        self.useCNN = useCNN
        self.data_save_loc = data_save_loc
        self.save_figs = save_figs
        self.show_figs = show_figs

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

    def pre(self): 
        print("starting image model")

        if str(type(self.data_file)) == "<class 'pandas.core.frame.DataFrame'>":
            self.df = self.data_file
        elif self.data_file[-4:] == ".csv":
            self.df = pd.read_csv(self.data_file)     
        
        # if statement fails, id is already index 
        try: 
            self.df = self.df.set_index(self.ID_dataset_col)
        except: 
            pass   

        features = list(self.feature_selection(self.df, self.target_vars, 10))

        # only use features determined by feature_selection in clinical data
        self.df = self.df[self.df.columns.intersection(features)]

        self.df.index.names = ["ID"]

        self.img_array = np.array([])
        matching_ids = []
        img_list = os.listdir(self.model_save_loc)

        # number of images that match proper resolution
        num_usable_img = 0

        # used for loading info
        imgs_processed = 0

        if self.load_numpy_img == True:
            self.img_array = np.load(os.path.join(self.img_array_save, os.listdir(self.img_array_save)[0]))
            if len(self.img_dimensions) == 3:
                flat_res = int((self.img_dimensions[0]*self.img_dimensions[1]*self.img_dimensions[2])+1)
            elif len(self.img_dimensions) == 2:
                flat_res = int((self.img_dimensions[0]*self.img_dimensions[1])+1)
            num_img = int(self.img_array.shape[0]/flat_res)
            self.img_array = np.reshape(self.img_array, (num_img, flat_res))

            ## retrieving ids
            img_df = pd.DataFrame(data=self.img_array)
            cols = list(img_df.columns)
            id_col = img_df[cols[-1]].tolist()
            dataset_id = self.df.index.tolist()

            # determine what to put first in loop
            if len(id_col) >= len(dataset_id):
                longest = id_col
                shortest = dataset_id
            elif len(dataset_id) > len(id_col):
                longest = dataset_id
                shortest = id_col

            for id in longest:
                for id2 in shortest:
                    if int(id) == int(id2):
                        matching_ids.append(id)

        elif self.load_numpy_img == False:

            for imgs in img_list:

                # find matching ids
                for ids in self.df.index:
                    ids = int(ids)
                    if ids == int(imgs[self.img_id_name_loc[0]:self.img_id_name_loc[1]]):
                        matching_ids.append(ids)
                        matching_ids = list(dict.fromkeys(matching_ids))

                # Collect/convert corresponding imagery
                print("starting data preparation process")
                for ids in matching_ids:
                    if ids == int(imgs[self.img_id_name_loc[0]:self.img_id_name_loc[1]]):
                        img = load_img(os.path.join(self.data_save_loc, imgs))
                        img_numpy_array = img_to_array(img)
                        if img_numpy_array.shape == self.img_dimensions:
                            img_numpy_array = img_numpy_array.flatten()
                            img_numpy_array = np.insert(img_numpy_array, len(img_numpy_array), ids)
                            num_usable_img = num_usable_img + 1
                            self.img_array = np.append(self.img_array, img_numpy_array, axis=0)
                            imgs_processed = imgs_processed + 1

                        else:
                            matching_ids.remove(ids)

                    ## Memory optimization
                    if psutil.virtual_memory().percent >= 50:
                        break

                    ## loading info
                    total_img = len(img_list)
                    percent_conv = (imgs_processed / total_img) * 100
                    print(str(round(percent_conv,2)) + " percent converted")
                    print(str(psutil.virtual_memory()))

            # save the array
            np.save(os.path.join(self.img_array_save, "img_array"), self.img_array)

            # reshape into legal dimensions
            self.img_array = np.reshape(self.img_array,(num_usable_img,int(self.img_array.size/num_usable_img)))

        self.df = self.df.loc[matching_ids]

        # initialize negative_vals as false
        negative_vals = False

        # determine activation function (relu or tanh) from if there are negative numbers in target variable
        df_values = self.df.values
        df_values = df_values.flatten()
        for val in df_values:
            val = float(val)
            if val < 0:
                negative_vals = True

        if negative_vals == True:
            self.activation_function = "tanh"
        else:
            self.activation_function = 'relu'

    def NN(self): 

        # initialize bool as false
        self.multiple_targets = False

        if str(type(self.target_vars)) == "<class 'list'>" and len(self.target_vars) > 1:
            self.multiple_targets = True

        self.pre()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Clinical
        # Get data
        df = self.df

        # y data
        labels = df[self.target_vars]
        # x data
        features = df.drop(self.target_vars, axis=1)

        X = features
        y = labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        # split test data into validation and test
        X_test, X_val = train_test_split(X_test, test_size=0.5, random_state=53)
        y_test, y_val = train_test_split(y_test, test_size=0.5, random_state=53)

        # normalize data
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)
        X_val = min_max_scaler.fit_transform(X_val)

        if self.multiple_targets:
            y_test = min_max_scaler.fit_transform(y_test)
            y_train = min_max_scaler.fit_transform(y_train)
            y_val = min_max_scaler.fit_transform(y_val)

        if str(type(y_train)) == "<class 'pandas.core.frame.DataFrame'>":
            y_train = y_train.to_numpy()

        if str(type(y_test)) == "<class 'pandas.core.frame.DataFrame'>":
            y_test = y_test.to_numpy()

        y_test = np.asarray(y_test).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        X_train = np.asarray(X_train).astype(np.float32)
        X_test = np.asarray(X_test).astype(np.float32)

        y_test = tf.convert_to_tensor(y_test)
        y_train = tf.convert_to_tensor(y_train)
        X_train = tf.convert_to_tensor(X_train)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Image

        X_train_img, X_test_img = train_test_split(self.img_array, test_size=0.4, random_state=42)

        X_test_img, X_val_img = train_test_split(X_test_img, test_size=0.5, random_state=34)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        def remove_ids(dataset):
            # initialize empty array
            newImg = np.empty((0, self.img_dimensions[0] * self.img_dimensions[1]))

            # remove ids from img data
            i = 0
            for arr in dataset:
                arr = np.delete(arr, -1)
                newImg = np.insert(newImg, i, arr, axis=0)
                i = i + 1

            return newImg

        if self.useCNN:
            X_train_img = remove_ids(X_train_img)

            X_test_img = remove_ids(X_test_img)

            X_val_img = remove_ids(X_val_img)

            # normalize data
            min_max_scaler = MinMaxScaler()
            X_train_img = min_max_scaler.fit_transform(X_train_img)
            X_test_img = min_max_scaler.fit_transform(X_test_img)
            X_val_img = min_max_scaler.fit_transform(X_val_img)

            X_train_img = np.reshape(X_train_img, (X_train_img.shape[0], self.img_dimensions[0], self.img_dimensions[1], 1))
            X_test_img = np.reshape(X_test_img, (X_test_img.shape[0], self.img_dimensions[0], self.img_dimensions[1], 1))
            X_val_img = np.reshape(X_val_img, (X_val_img.shape[0], self.img_dimensions[0], self.img_dimensions[1], 1))

            X_train = X_train_img
            X_test = X_test_img
            X_val = X_val_img

        if not self.useCNN:
            X_train_img = remove_ids(X_train_img)

            X_test_img = remove_ids(X_test_img)

            X_val_img = remove_ids(X_val_img)

            print(X_train.shape)
            print(X_train_img.shape)

            X_train = np.concatenate((X_train_img, X_train), axis=1)
            X_test = np.concatenate((X_test, X_test_img), axis=1)
            X_val = np.concatenate((X_val, X_val_img), axis=1)

            # normalize data
            min_max_scaler = MinMaxScaler()
            X_train = min_max_scaler.fit_transform(X_train)
            X_test = min_max_scaler.fit_transform(X_test)
            X_val = min_max_scaler.fit_transform(X_val)

        if self.multiple_targets:
            y_test = min_max_scaler.fit_transform(y_test)
            y_train = min_max_scaler.fit_transform(y_train)
            y_val = min_max_scaler.fit_transform(y_val)

        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        print(self.activation_function)

        if not self.load_fit:
            if not self.useCNN:
                if str(type(self.target_vars))!="<class 'list'>" or len(self.target_vars) == 1:

                    print(X_train.shape)

                    # set input shape to dimension of data
                    input = keras.layers.Input(shape=(X_train.shape[1],))

                    x = Dense(150, activation=self.activation_function)(input)
                    x = Dense(150, activation=self.activation_function)(x)
                    x = Dense(150, activation=self.activation_function)(x)
                    x = Dense(120, activation=self.activation_function)(x)
                    x = Dense(120, activation=self.activation_function)(x)
                    x = Dense(100, activation=self.activation_function)(x)
                    x = Dense(100, activation=self.activation_function)(x)
                    x = Dense(80, activation=self.activation_function)(x)
                    x = Dense(80, activation=self.activation_function)(x)
                    x = Dense(45, activation=self.activation_function)(x)
                    output = Dense(1, activation='linear')(x)
                    self.model = keras.Model(input, output)

                    self.model.compile(optimizer='adam',
                                      loss='mean_squared_error',
                                      metrics=['accuracy'])

                    self.fit = self.model.fit(X_train, y_train, epochs=self.epochs_num, batch_size=64)

                else:
                    input = keras.layers.Input(shape=(X_train.shape[1],))

                    x = Dense(150, activation=self.activation_function)(input)
                    x = Dense(150, activation=self.activation_function)(x)
                    x = Dense(150, activation=self.activation_function)(x)
                    x = Dense(120, activation=self.activation_function)(x)
                    x = Dense(120, activation=self.activation_function)(x)
                    x = Dense(100, activation=self.activation_function)(x)
                    x = Dense(100, activation=self.activation_function)(x)
                    x = Dense(80, activation=self.activation_function)(x)
                    x = Dense(80, activation=self.activation_function)(x)
                    x = Dense(45, activation=self.activation_function)(x)
                    output = Dense(len(self.target_vars), activation='linear')(x)

                    self.model = keras.Model(inputs=input, outputs=output)

                    self.model.compile(optimizer='adam',
                                  loss='mean_squared_error',
                                  metrics=['accuracy'])

                    self.fit = self.model.fit(X_train, y_train, epochs=self.epochs_num, batch_size=5)

            else:
                self.model = Sequential()

                self.model.add(layers.Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))
                self.model.add(layers.Activation('relu'))
                self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

                self.model.add(layers.Conv2D(64, (3, 3)))
                self.model.add(layers.Activation('relu'))
                self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

                self.model.add(layers.Flatten())

                self.model.add(layers.Dense(64))
                self.model.add(layers.Activation('relu'))

                self.model.add(layers.Dense(1))
                self.model.add(layers.Activation('linear'))

                self.model.compile(loss='mean_squared_error',
                              optimizer='adam',
                              metrics=['accuracy'])

                self.fit = self.model.fit(X_train, y_train, epochs=self.epochs_num)

        else:
            self.model = keras.models.load_model(self.model_save_loc)

        self.post()

    def post(self): 
        #plotting
        history = self.fit

        def plot(model_history, metric, graph_title):
            history = model_history
            plt.plot(history.history[metric])
            plt.title(graph_title)
            plt.ylabel(metric)
            plt.xlabel('epoch')

            save_path = os.path.join(self.data_save_loc, str(self.target_vars) + " " + metric + ".jpg")

            if "?" in save_path:
                save_path = save_path.replace("?", "")

            if self.save_figs == True:
                plt.savefig(save_path)

            if self.show_figs == True:
                plt.show()
            else:
                plt.clf()

        plot(history, 'loss', 'model loss')

        def save_fitted_model(model, save_location):
            model.save(save_location)

        if self.save_fit == True:
            save_fitted_model(self.model, self.model_save_loc)

        # utilize validation data
        prediction = self.model.predict(self.X_val, batch_size=1)

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

        percentAcc = self.percentageAccuracy(roundedPred, self.y_test)
        
        print("- - - - - - - - - - - - - Percentage Accuracy - - - - - - - - - - - - -")
        print(percentAcc)

        self.resultList.append(str(prediction))
        self.resultList.append(str(roundedPred))
        self.resultList.append(str(self.y_test))
        self.resultList.append(str(percentAcc))
