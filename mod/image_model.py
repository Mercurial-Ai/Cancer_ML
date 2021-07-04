from numpy.core.numeric import True_
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
import keras
from keras import layers
import os 
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import psutil
import matplotlib.pyplot as plt
from mod.percentage_accuracy import percentageAccuracy
import pydicom as dicom

class image_model: 
    def __init__(self, model_save_loc, data_file, target_vars, epochs_num, load_numpy_img, img_array_save, load_fit, save_fit, img_dimensions, img_id_name_loc, ID_dataset_col, useCNN, data_save_loc, save_figs, show_figs, load_dir): 
        self.model_save_loc = model_save_loc
        self.data_file = data_file 
        self.target_vars = target_vars
        self.epochs_num = epochs_num
        self.load_numpy_img = load_numpy_img
        self.img_array_save = img_array_save
        self.load_fit = load_fit
        self.save_fit = save_fit
        self.img_dimensions = img_dimensions
        self.img_id_name_loc = img_id_name_loc
        self.ID_dataset_col = ID_dataset_col
        self.useCNN = useCNN
        self.data_save_loc = data_save_loc
        self.save_figs = save_figs
        self.show_figs = show_figs
        self.load_dir = load_dir

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

    def checkBinary(self, target_var):

        orgPD = self.data_file
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

            if len(col) == 2:
                isBinary = True
            else: 
                isBinary = False

        return isBinary

    def get_y_distribution(self, y):
        
        y_len = len(y)

        counts_dict = y.value_counts().to_dict()
        
        percent_list = []
        for count in list(counts_dict.values()):
            percent = (count/y_len)*100
            percent = round(percent, 0)
            percent_list.append(percent)
        
        count_keys = list(counts_dict.keys())

        percent_dict = dict(zip(count_keys, percent_list))

        return percent_dict

    def collect_img_dirs(self, data_folder):
        img_directories = []

        for root, dirs, files, in os.walk(data_folder):
            for name in files:
                dir = os.path.join(root, name)
                img_directories.append(dir)

        return img_directories

    def pre(self):
        print("starting image model")

        self.isBinary = self.checkBinary(self.target_vars)

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

            print(self.img_array.shape)

            ## retrieving ids
            iter_list = []
            id_col = []
            for imgs in self.img_array:
                id = imgs[-1]
                id_col.append(id)
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

            for ids in matching_ids:
                if ids not in id_col:
                    id_col.remove(ids)

        elif self.load_numpy_img == False:

            img_list = self.collect_img_dirs(self.load_dir)

            # find matching ids
            index_id_list = []
            img_id_list = []
            for ids in self.df.index:
                index_id_list.append(ids)

            for imgs in img_list:

                img_id = int(imgs[self.img_id_name_loc[0]:self.img_id_name_loc[1]])
                img_id_list.append(img_id)

                set_index = set(index_id_list)
                matching_ids = list(set_index.intersection(img_id_list))

            print(len(img_list))
            print(len(matching_ids))

            for imgs in img_list:

                img = dicom.dcmread(imgs)
                img_id = img.PatientID

                for c in img_id:
                    if not c.isdigit():
                        img_id = img_id.replace(c,'')

                img_id = int(img_id)

                img_numpy_array = img.pixel_array
                if img_numpy_array.shape == self.img_dimensions and img_id in matching_ids:
                    img_numpy_array = img_numpy_array.flatten()
                    img_numpy_array = np.insert(img_numpy_array, len(img_numpy_array), img_id)
                    num_usable_img = num_usable_img + 1
                    self.img_array = np.append(self.img_array, img_numpy_array, axis=0)
                    imgs_processed = imgs_processed + 1

                    ## loading info
                    total_img = len(img_list)
                    percent_conv = (imgs_processed / total_img) * 100
                    print(str(round(percent_conv, 2)) + " percent converted")
                    print(str(psutil.virtual_memory()))

                ## Memory optimization
                if psutil.virtual_memory().percent >= 60:
                    break

            # reshape into legal dimensions
            self.img_array = np.reshape(self.img_array,(num_usable_img, int(self.img_array.size/num_usable_img)))

            # save the array
            np.save(os.path.join(self.img_array_save, "img_array"), self.img_array)

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

    def tensorboard(self): 
        if self.useCNN:
            model_name = "CNN"
        else: 
            model_name = "Image-Clinical"

        self.tb = TensorBoard(log_dir='logs/{}'.format(model_name))

    def NN(self): 
        self.tensorboard()

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

        # change y vals to 1, 2, 3, ...
        y_list = list(y)

        # remove duplicates to identify vals
        y_list = list(set(y_list))
        y_list.sort()

        i = 0
        binary_dict = {}
        for val in y_list:
            binary_dict[val] = i

            i = i + 1 

        i = 0
        new_y = pd.Series([])
        for val in y: 
            conv_y = binary_dict[val]

            new_y[i] = conv_y
            i = i + 1 

        y = new_y

        self.percent_dict = self.get_y_distribution(y)

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

        def remove_ids(img_array):

            new_shape = (img_array.shape[0], img_array.shape[1]-1)
            new_array = np.empty(shape=new_shape, dtype=np.int8)
            i = 0 
            for img in img_array:
                img = np.delete(img, -1)
                new_array[i] = img
                i = i + 1 

            return new_array

        if self.useCNN:
            X_train_img = remove_ids(X_train_img)

            X_test_img = remove_ids(X_test_img)

            X_val_img = remove_ids(X_val_img)

            # normalize data
            min_max_scaler = MinMaxScaler()
            
            i = 0
            new_array = np.empty(shape=(X_train_img.shape[0], self.img_dimensions[0], self.img_dimensions[1]), dtype=np.int8)
            for img in X_train_img:
                img = np.reshape(img, (self.img_dimensions[0], self.img_dimensions[1]))
                img = min_max_scaler.fit_transform(img)
                new_array[i] = img
                i = i + 1

            X_train_img = new_array
            
            i = 0
            new_array = np.empty(shape=(X_test_img.shape[0], self.img_dimensions[0], self.img_dimensions[1]), dtype=np.int8)
            for img in X_test_img:
                img = np.reshape(img, (self.img_dimensions[0], self.img_dimensions[1]))
                img = min_max_scaler.fit_transform(img)
                new_array[i] = img
                i = i + 1 

            X_test_img = new_array

            i = 0
            new_array = np.empty(shape=(X_val_img.shape[0], self.img_dimensions[0], self.img_dimensions[1]), dtype=np.int8)
            for img in X_val_img:
                img = np.reshape(img, (self.img_dimensions[0], self.img_dimensions[1]))
                img = min_max_scaler.fit_transform(img)
                new_array[i] = img
                i = i + 1 

            X_val_img = new_array

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

        # invert distribution dict to insert in class_weights
        val_list = list(self.percent_dict.values())
        val_list.reverse()

        key_list = list(self.percent_dict.keys())

        i = 0
        for val in val_list:
            val = val/100
            val_list[i] = val
            i = i + 1 

        self.percent_dict = dict(zip(key_list, val_list))
        print(self.percent_dict)

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

                    self.model.compile(optimizer='sgd',
                                      loss='mean_squared_error',
                                      metrics=['accuracy', MeanIoU(num_classes=2)])

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

                    self.model.compile(optimizer='sgd',
                                  loss='mean_squared_error',
                                  metrics=['accuracy'])

                    self.fit = self.model.fit(X_train, y_train, epochs=self.epochs_num, batch_size=5, class_weight=self.percent_dict)

            else:
                self.model = Sequential()

                self.model.add(layers.Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))
                self.model.add(layers.Activation('relu'))
                self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

                self.model.add(layers.Conv2D(64, (3, 3)))
                self.model.add(layers.Activation('relu'))
                self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

                self.model.add(layers.Flatten())

                self.model.add(layers.Dense(128))
                self.model.add(layers.Activation('relu'))

                self.model.add(layers.Dense(64))
                self.model.add(layers.Activation('relu'))

                self.model.add(layers.Dense(1))
                self.model.add(layers.Activation('linear'))

                self.model.compile(loss='mean_absolute_error',
                              optimizer='sgd',
                              metrics=['accuracy', MeanIoU(num_classes=2)])

                self.fit = self.model.fit(X_train, y_train, epochs=self.epochs_num, batch_size=32, callbacks=[self.tb], class_weight=self.percent_dict)

        else:
            self.model = keras.models.load_model(self.model_save_loc)

        self.post()

    def post(self): 

        iou_eval = MeanIoU(num_classes=2)

        #plotting
        history = self.fit

        def plot(model_history, metric, graph_title):
            history = model_history
            plt.plot(history.history[metric])
            plt.title(graph_title)
            plt.ylabel(metric)
            plt.xlabel('epoch')

            plt.show()

        plot(history, 'loss', 'model loss')
        plot(history, 'accuracy', 'accuracy')
        plot(history, 'mean_io_u', 'mean_iou')

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

        percentAcc = percentageAccuracy(roundedPred, self.y_val)

        print("- - - - - - - - - - - - - Percentage Accuracy - - - - - - - - - - - - -")
        print(percentAcc)

        iou_eval.update_state(prediction, self.y_val)
        print('mean iou: ' + str(iou_eval.result().numpy()))

        iou_eval.update_state(roundedPred, self.y_val)
        print('rounded pred mean iou: ' + str(iou_eval.result().numpy()))

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

        iou_eval.update_state(prediction, self.y_test)
        print('mean iou: ' + str(iou_eval.result().numpy()))

        iou_eval.update_state(roundedPred, self.y_test)
        print('rounded pred mean iou: ' + str(iou_eval.result().numpy()))

        self.resultList.append(str(prediction))
        self.resultList.append(str(roundedPred))
        self.resultList.append(str(self.y_test))
        self.resultList.append(str(percentAcc))
