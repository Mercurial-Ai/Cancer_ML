from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

from src.image_tools.import_numpy import import_numpy
from src.tokenize_dataset import tokenize_dataset

class data_pod:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.X_val = None

        self.y_train = None
        self.y_test = None
        self.y_val = None

class data_pipeline:

    def __init__(self, clinical_path, image_path, target):
        self.clinical_path = clinical_path
        self.image_path = image_path
        self.target = target

        self.only_clinical = data_pod()
        self.image_clinical = data_pod()
        self.image_only = data_pod()

    def load_data(self):
        self.df = pd.read_csv(self.clinical_path)
        self.clinical_ids = self.df[list(self.df.columns)[0]]

        self.df = self.df.set_index(list(self.df.columns)[0])

        self.img_array = import_numpy(self.image_path, self.clinical_ids)

        self.image_ids = self.img_array[:, -1]

        self.df = tokenize_dataset(self.df)

        # get patients in clinical data with ids that correspond with image ids
        self.filtered_df = self.df.loc[self.image_ids]

        self.concatenated_array = self.concatenate_image_clinical()

        self.partition_clinical_only_data()
        self.partition_image_clinical_data()
        self.partition_image_only_data()

    def concatenate_image_clinical(self):

        clinical_array = self.filtered_df.to_numpy()

        concatenated_array = np.concatenate((clinical_array, self.img_array), axis=1)

        return concatenated_array

    def split_data(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=84)

        # split test data into validation and test
        X_test, X_val = train_test_split(X_test, test_size=0.5, random_state=73)
        y_test, y_val = train_test_split(y_test, test_size=0.5, random_state=35)

        return X_train, X_test, y_train, y_test, X_val, y_val
    
    def partition_clinical_only_data(self):
        
        x = self.df.drop(self.target, axis=1)
        y = self.df[self.target]

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

    def partition_image_clinical_data(self):

        target_index = self.df.columns.get_loc(self.target)

        x = np.delete(self.concatenated_array, target_index, axis=1)
        y = self.concatenated_array[:, target_index]

        X_train, X_test, y_train, y_test, X_val, y_val = self.split_data(x, y)

        # normalize data
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)
        X_val = min_max_scaler.fit_transform(X_val)

        self.image_clinical.X_train = X_train
        self.image_clinical.X_test = X_test
        self.image_clinical.y_train = y_train
        self.image_clinical.y_test = y_test
        self.image_clinical.X_val = X_val
        self.image_clinical.y_val = y_val

    def partition_image_only_data(self):

        x = self.img_array
        y = self.filtered_df[self.target]

        X_train, X_test, y_train, y_test, X_val, y_val = self.split_data(x, y)

        # normalize data
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)
        X_val = min_max_scaler.fit_transform(X_val)

        self.image_only.X_train = X_train
        self.image_only.X_test = X_test
        self.image_only.y_train = y_train
        self.image_only.y_test = y_test
        self.image_only.X_val = X_val
        self.image_only.y_val = y_val
