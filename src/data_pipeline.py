from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import math
from src.image_tools.import_numpy import import_numpy
from src.tokenize_dataset import tokenize_dataset
from src.balance_y_classes import balance_y_classes

class data_pod:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.X_val = None

        self.y_train = None
        self.y_test = None
        self.y_val = None

class data_pipeline:

    def __init__(self, clinical_path, image_features_path, image_path, target):
        self.clinical_path = clinical_path
        self.image_features_path = image_features_path
        self.image_path = image_path
        self.target = target

        self.only_clinical = data_pod()
        self.image_clinical = data_pod()
        self.image_only = data_pod()

    def load_data(self):
        self.df = pd.read_csv(self.clinical_path, low_memory=False)
        if self.image_features_path != None:
            self.image_features = pd.read_csv(self.image_features_path, low_memory=False)

            # drop ids from image features to prevent duplicates
            self.image_features = self.image_features.drop("Patient ID", axis=1)

        self.clinical_ids = self.df[list(self.df.columns)[0]]

        if self.image_features_path != None:
            self.df = pd.concat([self.df, self.image_features], axis=1)

        self.df = self.df.set_index(str(list(self.df.columns)[0]))

        self.df = tokenize_dataset(self.df)

        # if image path = None, dataset should be clinical only and imagery does not need to be imported
        if self.image_path != None:
            self.img_array = import_numpy(self.image_path, self.clinical_ids)

            self.slice_data()

            self.image_ids = self.img_array[:, -1]

            self.train_ids, self.test_ids = train_test_split(self.image_ids, test_size=0.2, random_state=84)

            self.test_ids, self.val_ids = train_test_split(self.test_ids, test_size=0.2, random_state=84)

            # remove ids from img_array
            self.img_array = np.delete(self.img_array, -1, axis=1)

            # get patients in clinical data with ids that correspond with image ids
            self.filtered_df = self.df.loc[self.image_ids]

            self.partition_image_clinical_data()
            self.partition_image_only_data()

        self.partition_clinical_only_data()

    def concatenate_image_clinical(self, clinical_array):

        concatenated_array = np.concatenate((clinical_array, self.img_array), axis=1)

        return concatenated_array

    def split_data(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=84)

        # split test data into validation and test
        X_test, X_val = train_test_split(X_test, test_size=0.5, random_state=84)
        y_test, y_val = train_test_split(y_test, test_size=0.5, random_state=84)

        return X_train, X_test, y_train, y_test, X_val, y_val
    
    def partition_clinical_only_data(self):
        
        if type(self.target) == list:
            for var in self.target:
                var = var.replace("\n", "")
                x = self.df.drop(var, axis=1)
        else:
            x = self.df.drop(self.target, axis=1)
        
        y = self.df[self.target]

        # check if METABRIC
        if y.shape[-1] == 7:
            x, y = balance_y_classes(x, y, 1, shuffle_data=True)
        else:
            x, y = balance_y_classes(x, y, 11, shuffle_data=True)

        X_train, X_test, y_train, y_test, X_val, y_val = self.split_data(x, y)

        # normalize data
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)
        X_val = min_max_scaler.fit_transform(X_val)

        self.only_clinical.X_train = X_train
        self.only_clinical.X_test = X_test
        self.only_clinical.y_train = y_train
        self.only_clinical.y_test = y_test
        self.only_clinical.X_val = X_val
        self.only_clinical.y_val = y_val

    def split_modalities(self, x):

        clinical_x = x[:, :self.clinical_x.shape[1]]
        image_x = x[:, self.clinical_x.shape[1]:]

        # unflatten images in image_x
        unflattened_array = np.empty(shape=(image_x.shape[0], int(math.sqrt(image_x.shape[-1])), int(math.sqrt(image_x.shape[-1])), 1), dtype=np.int8)
        i = 0
        for image in image_x:
            image = np.reshape(image, (1, 512, 512, 1))
            unflattened_array[i] = image

            i = i + 1

        return clinical_x, unflattened_array

    def partition_image_clinical_data(self):

        self.clinical_x = self.filtered_df

        if type(self.target) == list:
            for var in self.target:
                var = var.replace("\n", "")
                self.clinical_x = self.clinical_x.drop(var, axis=1)
        else:
            self.clinical_x = self.clinical_x.drop(self.target, axis=1)

        y = self.filtered_df[self.target]

        x = self.concatenate_image_clinical(self.clinical_x)

        x, y = balance_y_classes(x, y, 11, shuffle_data=True)

        X_train, X_test, y_train, y_test, X_val, y_val = self.split_data(x, y)

        # normalize data
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)
        X_val = min_max_scaler.fit_transform(X_val)

        X_train = [self.split_modalities(X_train)]
        X_test = [self.split_modalities(X_test)]
        X_val = [self.split_modalities(X_val)]

        self.image_clinical.X_train = X_train
        self.image_clinical.X_test = X_test
        self.image_clinical.y_train = y_train
        self.image_clinical.y_test = y_test
        self.image_clinical.X_val = X_val
        self.image_clinical.y_val = y_val

    def partition_image_only_data(self):

        x = self.img_array
        y = self.filtered_df[self.target]

        x, y = balance_y_classes(x, y, 11, shuffle_data=True)

        X_train, X_test, y_train, y_test, X_val, y_val = self.split_data(x, y)

        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)
        X_val = min_max_scaler.fit_transform(X_val)

        # reshape back into 2d images
        X_train = np.reshape(X_train, (X_train.shape[0], int(math.sqrt(X_train.shape[1])), int(math.sqrt(X_train.shape[1]))))
        X_test = np.reshape(X_test, (X_test.shape[0], int(math.sqrt(X_test.shape[1])), int(math.sqrt(X_test.shape[1]))))
        X_val = np.reshape(X_val, (X_val.shape[0], int(math.sqrt(X_val.shape[1])), int(math.sqrt(X_val.shape[1]))))

        # add additional dimension at the end of the shape to each partition
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)

        self.image_only.X_train = X_train
        self.image_only.X_test = X_test
        self.image_only.X_val = X_val
        self.image_only.y_train = y_train
        self.image_only.y_test = y_test
        self.image_only.y_val = y_val

    def slice_data(self):
        slice_size = 0.25

        self.img_array = self.img_array[0:int(round(self.img_array.shape[0]*slice_size, 0))]

        print("img_array shape:", self.img_array.shape)
