from matplotlib import scale
from src.PeakCluster import PeakCluster
from src.data_pipeline import data_pipeline
from src.clinical_only import clinical_only
from src.image_model import image_model
from src.cnn import cnn
from collections import Counter
import math
import pickle
from src.isolation_forest import isolation_forest
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from src.tabular_k_means import tabular_k_means
import tensorflow as tf
import torch

class cancer_ml:

    def tuple_to_list(self, t):
        return list(map(self.tuple_to_list, t)) if isinstance(t, (list, tuple)) else t

    def __init__(self, dataset, target, model="clinical_only", crop_size=(256, 256)):
        self.dataset = dataset
        self.target = target
        self.model = model
        self.crop_size = crop_size

        if self.dataset == "duke":
            self.collect_duke()
        elif self.dataset == "metabric":
            self.collect_METABRIC()

        # initialize model bools
        self.clinical = False
        self.image_clinical = False
        self.cnn = False

        if self.model == "clinical_only":
            self.clinical = True
            
            self.data_pipe.only_clinical.X_train, self.data_pipe.only_clinical.y_train = self.remove_outliers(self.data_pipe.only_clinical.X_train, self.data_pipe.only_clinical.y_train)

            self.data_pipe.only_clinical.X_train, self.data_pipe.only_clinical.y_train = tabular_k_means(self.data_pipe.only_clinical.X_train, self.data_pipe.only_clinical.y_train)

        elif self.model == "image_clinical":
            self.image_clinical = True

            prev_indices = list(self.data_pipe.image_clinical.y_train.index)
            self.data_pipe.image_clinical.X_train[1], self.data_pipe.image_clinical.y_train = self.remove_outliers(self.data_pipe.image_clinical.X_train[1], self.data_pipe.image_clinical.y_train)

            removed_indices = []
            j = 0
            for i in prev_indices:
                if i not in list(self.data_pipe.image_clinical.y_train.index):
                    removed_indices.append(j)

                j = j + 1

            self.data_pipe.image_clinical.X_train[0] = np.delete(self.data_pipe.image_clinical.X_train[0], removed_indices, 0)

            self.data_pipe.image_clinical.X_train[0] = tf.image.random_crop(self.data_pipe.image_clinical.X_train[0], (self.data_pipe.image_clinical.X_train[0].shape[0], self.data_pipe.image_clinical.X_train[0].shape[1], self.crop_size[0], self.crop_size[1]))
            self.data_pipe.image_clinical.X_test[0] = tf.image.random_crop(self.data_pipe.image_clinical.X_test[0], (self.data_pipe.image_clinical.X_test[0].shape[0], self.data_pipe.image_clinical.X_test[0].shape[1], self.crop_size[0], self.crop_size[1]))
            self.data_pipe.image_clinical.X_val[0] = tf.image.random_crop(self.data_pipe.image_clinical.X_val[0], (self.data_pipe.image_clinical.X_val[0].shape[0], self.data_pipe.image_clinical.X_val[0].shape[1], self.crop_size[0], self.crop_size[1]))

        elif self.model == "cnn":
            self.cnn = True

    def collect_duke(self):

        if self.model !="clinical_only":
            self.data_pipe = data_pipeline("data/Duke-Breast-Cancer-MRI/Clinical and Other Features (edited).csv", "data/Duke-Breast-Cancer-MRI/Imaging_Features.csv", "/scratch/data/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI", self.target)
        else: 
            self.data_pipe = data_pipeline("data/Duke-Breast-Cancer-MRI/Clinical and Other Features (edited).csv", None, None, self.target)

        self.data_pipe.load_data()

    def collect_METABRIC(self):
        self.data_pipe = data_pipeline("data/METABRIC_RNA_Mutation/METABRIC_RNA_Mutation.csv", None, None, self.target)
        self.data_pipe.load_data()

    def run_model(self):

        if self.clinical:
            self.model = clinical_only(load_model=False)
            self.model.get_model(self.data_pipe.only_clinical.X_train, self.data_pipe.only_clinical.y_train, self.data_pipe.only_clinical.X_val, self.data_pipe.only_clinical.y_val)
        elif self.image_clinical:
            self.model = image_model(load_model=False)
            if type(self.data_pipe.image_clinical.X_train) != torch.Tensor:
                i = 0
                for arr in self.data_pipe.image_clinical.X_train:
                    if type(arr) != torch.Tensor:
                        arr = torch.from_numpy(np.array(arr))
                        self.data_pipe.image_clinical.X_train[i] = arr

                    i = i + 1

            if type(self.data_pipe.image_clinical.X_val) != torch.Tensor:
                i = 0
                for arr in self.data_pipe.image_clinical.X_val:
                    if type(arr) != torch.Tensor:
                        arr = torch.from_numpy(np.array(arr))
                        self.data_pipe.image_clinical.X_val[i] = arr

                    i = i + 1
            if type(self.data_pipe.image_clinical.y_train) != torch.Tensor:
                self.data_pipe.image_clinical.y_train = torch.from_numpy(np.array(self.data_pipe.image_clinical.y_train))
            if type(self.data_pipe.image_clinical.y_val) != torch.Tensor:
                self.data_pipe.image_clinical.y_val = torch.from_numpy(np.array(self.data_pipe.image_clinical.y_val))
            self.model.get_model(self.data_pipe.image_clinical.X_train, self.data_pipe.image_clinical.y_train, self.data_pipe.image_clinical.X_val, self.data_pipe.image_clinical.y_val)
        elif self.cnn:
            self.model = cnn(load_model=False)
            if type(self.data_pipe.image_only.X_train) != torch.Tensor:
                self.data_pipe.image_only.X_train = torch.from_numpy(np.expand_dims(self.data_pipe.image_only.X_train, -1))
            if type(self.data_pipe.image_only.y_train) != torch.Tensor:
                self.data_pipe.image_only.y_train = torch.from_numpy(np.array(self.data_pipe.image_only.y_train))
            if type(self.data_pipe.image_only.X_val) != torch.Tensor:
                self.data_pipe.image_only.X_val = torch.from_numpy(np.expand_dims(self.data_pipe.image_only.X_val, -1))
            if type(self.data_pipe.image_only.y_val) != torch.Tensor:
                self.data_pipe.image_only.y_val = torch.from_numpy(np.array(self.data_pipe.image_only.y_val))
            self.model.get_model(self.data_pipe.image_only.X_train, self.data_pipe.image_only.y_train, self.data_pipe.image_only.X_val, self.data_pipe.image_only.y_val)

    def test_model(self):
        print("Testing model")
        
        if self.clinical:
            print(self.model.test_model(self.data_pipe.only_clinical.X_test, self.data_pipe.only_clinical.y_test))
        elif self.image_clinical:
            print(self.model.test_model([self.data_pipe.image_clinical.X_test[0], self.data_pipe.image_clinical.X_test[1]], self.data_pipe.image_clinical.y_test))
        elif self.cnn:
            print(self.model.test_model(self.data_pipe.image_only.X_test, self.data_pipe.image_only.y_test))

    def remove_outliers(self, X, y):

        predicted = isolation_forest(X, y)
        
        self.non_outlier_indices = []
        i = 0
        for prediction in predicted:
            if prediction != -1:
                self.non_outlier_indices.append(i)

            i = i + 1

        num_outliers = len(predicted) - len(self.non_outlier_indices)

        print("Num Outliers:", num_outliers)

        if type(X) == tuple or type(X) == list:
            X = list(X)
            i = 0
            for array in X:
                array = array[self.non_outlier_indices, :]
                X[i] = array

                i = i + 1

        else:
            if str(type(X)) == "<class 'numpy.ndarray'>":
                X = X[self.non_outlier_indices]
            else:
                X = X.iloc[self.non_outlier_indices]

        if str(type(y)) == "<class 'numpy.ndarray'>":
            y = y[self.non_outlier_indices]
        else:
            y = y.iloc[self.non_outlier_indices]

        return X, y

    def setup_cluster(self):
        X = self.data_pipe.image_only.X_train

        X = tf.image.random_crop(value=X, size=(X.shape[0], X.shape[1], self.crop_size[0], self.crop_size[1]))
        
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]*X.shape[3]))
        self.model = PeakCluster(X)

        # determine number of each label
        self.label_counts = dict(Counter(self.model.labels_))

    def make_class_inference(self, image_array):

        image_array = image_array.flatten()
        image_array = np.expand_dims(image_array, axis=0)

        inference = self.neigh.predict(image_array)

        return inference

    def k_neighbors(self):

        X = self.data_pipe.image_only.X_train
        X_test = self.data_pipe.image_only.X_test
        X_val = self.data_pipe.image_only.X_val

        X = tf.image.random_crop(X, (X.shape[0], X.shape[1], self.crop_size[0], self.crop_size[1]))
        X_test = tf.image.random_crop(X_test, (X_test.shape[0], X_test.shape[1], self.crop_size[0], self.crop_size[1]))
        X_val = tf.image.random_crop(X_val, (X_val.shape[0], X_val.shape[1], self.crop_size[0], self.crop_size[1]))

        self.data_pipe.image_only.X_train = X
        self.data_pipe.image_only.X_test = X_test
        self.data_pipe.image_only.X_val = X_val

        x_shape_train = X.shape
        x_shape_test = X_test.shape
        x_shape_val = X_val.shape

        # flatten X for KNeighbors
        k_X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]*X.shape[3])).astype('float16')

        # adjust y for x format
        def y_2d_to_1d(pre_x_shape, y):
            new_y = np.empty(pre_x_shape[0]*pre_x_shape[1])
            j = 0
            for i in range(pre_x_shape[1]):
                new_y[j:j+pre_x_shape[0]] = y

                j = j + pre_x_shape[0]
            
            return new_y

        self.data_pipe.image_only.y_train = y_2d_to_1d(x_shape_train, self.data_pipe.image_only.y_train)
        self.data_pipe.image_only.y_test = y_2d_to_1d(x_shape_test, self.data_pipe.image_only.y_test)
        self.data_pipe.image_only.y_val = y_2d_to_1d(x_shape_val, self.data_pipe.image_only.y_val)

        n_clusters = len(list(set(self.model.labels_)))

        self.neigh = KNeighborsClassifier(n_neighbors=n_clusters)
        self.neigh.fit(k_X, self.model.labels_)

        X = np.array(X)
        X_test = np.array(X_test)
        X_val = np.array(X_val)

        print(X.shape)

        self.all_indices_train = []
        i = 0
        for p in X:
            p = self.equalize_classes(p)
            self.all_indices_train.append(self.collected_indices_train)
            X[i] = p
            i = i + 1
        self.all_indices_test = []
        i = 0
        for p in X_test:
            p = self.equalize_test(p)
            self.all_indices_test.append(self.collected_indices_test)
            X[i] = p
            i = i + 1
        self.all_indices_val = []
        i = 0
        for p in X_val:
            p = self.equalize_val(p)
            self.all_indices_val.append(self.collected_indices_val)
            X[i] = p
            i = i + 1

        self.equalize_image_clinical()

    def get_classes(self):
        return self.model.labels_

    def scale_clinical(self, arr, scalar):

        new_array = np.empty((arr.shape[0]*scalar, arr.shape[-1]))
        k = 0
        for i in range(arr.shape[0]):
            for j in range(scalar):
                arr_slice = arr[i]
                arr_slice = np.expand_dims(arr_slice, 0)
                new_array[k] = arr_slice

                k = k + 1

        return new_array

    def equalize_image_clinical(self):

        self.data_pipe.image_clinical.X_train[1] = self.scale_clinical(self.data_pipe.image_clinical.X_train[1], self.pre_x_shape_train[1])
        self.data_pipe.image_clinical.X_test[1] = self.scale_clinical(self.data_pipe.image_clinical.X_test[1], self.pre_x_shape_test[1])
        self.data_pipe.image_clinical.X_val[1] = self.scale_clinical(self.data_pipe.image_clinical.X_val[1], self.pre_x_shape_val[1])

        i = 0
        for p in self.data_pipe.image_clinical.X_train[0]:
            p = p[self.all_indices_train[i]]
            self.data_pipe.image_clinical.X_train[0][i] = p

            i = i + 1

        i = 0
        for p in self.data_pipe.image_clinical.X_test[0]:
            p = p[self.all_indices_test[i]]
            self.data_pipe.image_clinical.X_test[0][i] = p

            i = i + 1

        i = 0
        for p in self.data_pipe.image_clinical.X_val[0]:
            p = p[self.all_indices_val[i]]
            self.data_pipe.image_clinical.X_val[0][i] = p

            i = i + 1

        self.data_pipe.image_clinical.y_train = self.scale_clinical(np.expand_dims(self.data_pipe.image_clinical.y_train, -1), self.pre_x_shape_train[1])
        self.data_pipe.image_clinical.y_test = self.scale_clinical(np.expand_dims(self.data_pipe.image_clinical.y_test, -1), self.pre_x_shape_test[1])
        self.data_pipe.image_clinical.y_val = self.scale_clinical(np.expand_dims(self.data_pipe.image_clinical.y_val, -1), self.pre_x_shape_val[1])

        self.data_pipe.image_clinical.y_train = self.data_pipe.image_clinical.y_train[self.collected_indices_train]
        self.data_pipe.image_clinical.y_test = self.data_pipe.image_clinical.y_test[self.collected_indices_test]
        self.data_pipe.image_clinical.y_val = self.data_pipe.image_clinical.y_val[self.collected_indices_val]

    def equalize_test(self, img_array):

        y_test = self.data_pipe.image_only.y_test

        if str(type(y_test)) == "<class 'pandas.core.series.Series'>":
            y_test = y_test.to_numpy()

        # dictionary containing image data per label
        class_array_dict = {}

        # dictionary containing y data per label
        class_y_dict = {}

        inferences = []
        for image in img_array:
            inference = self.make_class_inference(image)[0]
            inferences.append(inference)

        label_counts = dict(Counter(inferences))
        
        num_clusters_used = 0
        for label in list(label_counts.keys()):

            count = label_counts[label]

            if count != 1:
                # list of indices utilized to collect y
                collected_indices = []

                class_array = np.empty(shape=(label_counts[label], img_array.shape[1], img_array.shape[2]), dtype=np.int8)
            
                i = 0
                j = 0
                for image in img_array:

                    inference = inferences[i]

                    if inference == label:
                        image = np.squeeze(image)
                        class_array[j] = image
                        collected_indices.append(i)
                        j = j + 1

                    i = i + 1
                        
                class_array_dict[label] = class_array 

                if type(y_test) == pd.DataFrame:
                    y = y_test.iloc[collected_indices]
                else:
                    y = y_test[collected_indices]

                class_y_dict[label] = y

                num_clusters_used = num_clusters_used + 1

        self.collected_indices_test = collected_indices

        # find lowest count of labels excluding labels with only 1 instance
        labels = list(label_counts.keys())

        filtered_labels = []
        for label in labels:
            if self.label_counts[label] != 1:
                filtered_labels.append(label_counts[label])
        
        lowest_count = min(filtered_labels)

        for label in list(class_array_dict.keys()):

            class_array = class_array_dict[label]
            y_array = class_y_dict[label]

            new_array = class_array[:lowest_count]
            new_y = y_array[:lowest_count]

            class_array_dict[label] = new_array
            class_y_dict[label] = new_y

        arrays = []
        for data in list(class_y_dict.values()):
            arrays.append(data)

        new_y = np.concatenate(tuple(arrays), axis=0)

        self.data_pipe.image_only.y_test = new_y.astype('int8')

        new_data = np.empty(shape=(lowest_count*num_clusters_used, self.crop_size[0], self.crop_size[1]), dtype=np.int8)
        i = 0
        for image_array in list(class_array_dict.values()):

            for image in image_array:
                new_data[i] = image

                i = i + 1

        new_data = np.expand_dims(new_data, axis=-1)

        return new_data

    def equalize_val(self, img_array):

        y_val = self.data_pipe.image_only.y_val

        if str(type(y_val)) == "<class 'pandas.core.series.Series'>":
            y_val = y_val.to_numpy()

        # dictionary containing image data per label
        class_array_dict = {}

        # dictionary containing y data per label
        class_y_dict = {}

        inferences = []
        for image in img_array:
            inference = self.make_class_inference(image)[0]
            inferences.append(inference)

        label_counts = dict(Counter(inferences))
        
        num_clusters_used = 0
        for label in list(label_counts.keys()):

            count = label_counts[label]

            if count != 1:
                # list of indices utilized to collect y
                collected_indices = []

                class_array = np.empty(shape=(label_counts[label], img_array.shape[1], img_array.shape[2]), dtype=np.int8)
            
                i = 0
                j = 0
                for image in img_array:

                    inference = inferences[i]

                    if inference == label:
                        image = np.squeeze(image)
                        class_array[j] = image
                        collected_indices.append(i)
                        j = j + 1

                    i = i + 1
                        
                class_array_dict[label] = class_array 

                if type(y_val) == pd.DataFrame:
                    y = y_val.iloc[collected_indices]
                else:
                    y = y_val[collected_indices]

                class_y_dict[label] = y

                num_clusters_used = num_clusters_used + 1

        self.collected_indices_val = collected_indices

        # find lowest count of labels excluding labels with only 1 instance
        labels = list(label_counts.keys())

        filtered_labels = []
        for label in labels:
            if self.label_counts[label] != 1:
                filtered_labels.append(label_counts[label])
        
        lowest_count = min(filtered_labels)

        for label in list(class_array_dict.keys()):

            class_array = class_array_dict[label]
            y_array = class_y_dict[label]

            new_array = class_array[:lowest_count]
            new_y = y_array[:lowest_count]

            class_array_dict[label] = new_array
            class_y_dict[label] = new_y

        arrays = []
        for data in list(class_y_dict.values()):
            arrays.append(data)

        new_y = np.concatenate(tuple(arrays), axis=0)

        self.data_pipe.image_only.y_val = new_y.astype('int8')

        new_data = np.empty(shape=(lowest_count*num_clusters_used, self.crop_size[0], self.crop_size[1]), dtype=np.int8)
        i = 0
        for image_array in list(class_array_dict.values()):

            for image in image_array:
                new_data[i] = image

                i = i + 1

        new_data = np.expand_dims(new_data, axis=-1)

        return new_data

    def divide_into_classes(self, image_array, n_clusters):

        y_train = self.data_pipe.image_only.y_train

        if str(type(y_train)) == "<class 'pandas.core.series.Series'>":
            y_train = y_train.to_numpy()

        # dictionary containing image data per label
        class_array_dict = {}

        # dictionary containing y data per label
        class_y_dict = {}

        self.num_clusters_used = 0
        for label in range(n_clusters):

            if self.label_counts[label] != 1:

                # list of indices utilized to collect y
                collected_indices = []

                class_array = np.empty(shape=(self.label_counts[label], image_array.shape[1], image_array.shape[2]), dtype=np.int8)
            
                i = 0
                j = 0
                for image in image_array:

                    inference = self.model.labels_[i]

                    if inference == label:
                        class_array[j] = image
                        collected_indices.append(i)
                        j = j + 1

                    i = i + 1
                        
                class_array_dict[label] = class_array

                if type(y_train) == pd.DataFrame:
                    y = y_train.iloc[collected_indices]
                else:
                    y = y_train[collected_indices]

                class_y_dict[label] = y

                self.num_clusters_used = self.num_clusters_used + 1

        self.collected_indices_train = collected_indices

        return class_array_dict, class_y_dict

    def equalize_classes(self, image_array):

        # find lowest count of labels excluding labels with only 1 instance
        labels = list(self.label_counts.keys())

        filtered_labels = []
        for label in labels:
            if self.label_counts[label] != 1:
                filtered_labels.append(self.label_counts[label])
        
        lowest_count = min(filtered_labels)

        n_clusters = len(self.label_counts)
        class_array_dict, class_y_dict = self.divide_into_classes(image_array, n_clusters)

        label_indices_removed = []
        nums_saved = []
        for label in list(class_array_dict.keys()):

            class_array = class_array_dict[label]
            y_array = class_y_dict[label]

            new_array = class_array[:lowest_count]
            new_y = y_array[:lowest_count]

            num_saved = len(new_array)
            nums_saved.append(num_saved)

            indices_removed = list(range(num_saved, len(class_array[lowest_count:])))
            label_indices_removed.append(indices_removed)

            class_array_dict[label] = new_array
            class_y_dict[label] = new_y
                
        # flatten nested list
        new_label_indices_removed = []
        for nested in label_indices_removed:
            for element in nested:
                new_label_indices_removed.append(element)

        indices_removed = new_label_indices_removed

        arrays = []
        for data in list(class_y_dict.values()):
            arrays.append(data)

        new_y = np.concatenate(tuple(arrays), axis=0)

        self.data_pipe.image_only.y_train = new_y.astype('int8')

        new_data = np.empty(shape=(lowest_count*self.num_clusters_used, self.crop_size[0], self.crop_size[1]), dtype=np.int8)
        i = 0
        for image_array in list(class_array_dict.values()):

            for image in image_array:
                new_data[i] = image

                i = i + 1

        new_data = np.expand_dims(new_data, axis=-1)

        return new_data

    def save_arrays(self):
        imageFile = open('image_only.pickle', 'w+b')
        pickle.dump(self.data_pipe.image_only, imageFile, protocol=4)
        imageFile.close()

        ICfile = open('image_clinical.pickle', 'w+b')
        pickle.dump(self.data_pipe.image_clinical, ICfile, protocol=4)
        ICfile.close()

        clinicalFile = open('clinical_only.pickle', 'w+b')
        pickle.dump(self.data_pipe.only_clinical, clinicalFile, protocol=4)
        clinicalFile.close()

    def load_arrays(self):
        imageFile = open('image_only.pickle', 'r+b')
        self.data_pipe.image_only = pickle.load(imageFile)
        imageFile.close()

        ICfile = open('image_clinical.pickle', 'r+b')
        self.data_pipe.image_clinical = pickle.load(ICfile)
        ICfile.close()

        clinicalFile = open('clinical_only.pickle', 'r+b')
        self.data_pipe.only_clinical = pickle.load(clinicalFile)
        clinicalFile.close()
        