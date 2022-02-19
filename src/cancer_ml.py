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
from src.random_crop import random_crop

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
        elif self.dataset == "hn1":
            self.collect_HN1()
        elif self.dataset == "metabric":
            self.collect_METABRIC()

        # initialize model bools
        self.clinical = False
        self.image_clinical = False
        self.cnn = False

        if self.model == "clinical_only":
            self.clinical = True
            
            self.data_pipe.only_clinical.X_train, self.data_pipe.only_clinical.y_train = self.remove_outliers(self.data_pipe.only_clinical.X_train, self.data_pipe.only_clinical.y_train)

        elif self.model == "image_clinical":
            self.image_clinical = True

            self.data_pipe.image_clinical.X_train = self.tuple_to_list(self.data_pipe.image_clinical.X_train)
            self.data_pipe.image_clinical.X_test = self.tuple_to_list(self.data_pipe.image_clinical.X_test)
            self.data_pipe.image_clinical.X_val = self.tuple_to_list(self.data_pipe.image_clinical.X_val)

            self.data_pipe.image_clinical.X_train[0], self.data_pipe.image_clinical.y_train = self.remove_outliers(self.data_pipe.image_clinical.X_train[0], self.data_pipe.image_clinical.y_train)

            self.data_pipe.image_clinical.X_train[0][1] = random_crop(list(self.data_pipe.image_clinical.X_train[0][1]), (self.crop_size[0], self.crop_size[1], 1))
            self.data_pipe.image_clinical.X_test[0][1] = random_crop(list(self.data_pipe.image_clinical.X_test[0][1]), (self.crop_size[0], self.crop_size[1], 1))
            self.data_pipe.image_clinical.X_val[0][1] = random_crop(list(self.data_pipe.image_clinical.X_val[0][1]), (self.crop_size[0], self.crop_size[1], 1))

            self.data_pipe.image_clinical.X_train[0][1] = self.data_pipe.image_clinical.X_train[0][1][self.non_outlier_indices]

        elif self.model == "cnn":
            self.cnn = True

            self.setup_cluster()
            self.k_neighbors()

    def collect_duke(self):

        if self.model !="clinical_only":
            self.data_pipe = data_pipeline("data/Duke-Breast-Cancer-MRI/Clinical and Other Features (edited).csv", "data/Duke-Breast-Cancer-MRI/img_array_duke.npy", self.target)
        else: 
            self.data_pipe = data_pipeline("data/Duke-Breast-Cancer-MRI/Clinical and Other Features (edited).csv", None, self.target)

        self.data_pipe.load_data()
    
    def collect_HN1(self):
        self.data_pipe = data_pipeline("data/HNSCC-HN1/Copy of HEAD-NECK-RADIOMICS-HN1 Clinical data updated July 2020 (adjusted chemotherapy var).csv", "data/HNSCC-HN1/img_array.npy", self.target)
        self.data_pipe.load_data()

    def collect_METABRIC(self):
        self.data_pipe = data_pipeline("data/METABRIC_RNA_Mutation/METABRIC_RNA_Mutation.csv", None, self.target)
        self.data_pipe.load_data()

    def run_model(self):

        if self.clinical:
            self.model = clinical_only(load_model=False)
            self.model.get_model(self.data_pipe.only_clinical.X_train, self.data_pipe.only_clinical.y_train, self.data_pipe.only_clinical.X_val, self.data_pipe.only_clinical.y_val)
        elif self.image_clinical:
            self.model = image_model(load_model=False)
            self.model.get_model(self.data_pipe.image_clinical.X_train, self.data_pipe.image_clinical.y_train, self.data_pipe.image_clinical.X_val, self.data_pipe.image_clinical.y_val)
        elif self.cnn:
            self.model = cnn(load_model=False)
            self.model.get_model(self.data_pipe.image_only.X_train, self.data_pipe.image_only.y_train, self.data_pipe.image_only.X_val, self.data_pipe.image_only.y_val)

    def test_model(self):
        
        if self.clinical:
            print(self.model.test_model(self.data_pipe.only_clinical.X_test, self.data_pipe.only_clinical.y_test))
        elif self.image_clinical:
            print(self.model.test_model(self.data_pipe.image_clinical.X_test, self.data_pipe.image_clinical.y_test))
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

        X = random_crop(X, (self.crop_size[0], self.crop_size[1], 1))   

        X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
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

        X = random_crop(X, (self.crop_size[0], self.crop_size[1], 1))
        X_test = random_crop(X_test, (self.crop_size[0], self.crop_size[1], 1))
        X_val = random_crop(X_val, (self.crop_size[0], self.crop_size[1], 1))

        self.data_pipe.image_only.X_test = X_test
        self.data_pipe.image_only.X_val = X_val

        # flatten X for KNeighbors
        X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2])).astype('float32')

        n_clusters = len(list(set(self.model.labels_)))

        self.neigh = KNeighborsClassifier(n_neighbors=n_clusters)
        self.neigh.fit(X, self.model.labels_)

        # unflatten X
        X = np.reshape(X, (-1, int(math.sqrt(X.shape[1])), int(math.sqrt(X.shape[1]))))

        self.data_pipe.image_only.X_train = self.equalize_classes(X)
        self.data_pipe.image_only.X_test = self.equalize_test(X_test)
        self.data_pipe.image_only.X_val = self.equalize_val(X_val)

        self.equalize_image_clinical()

    def get_classes(self):
        return self.model.labels_

    def equalize_image_clinical(self):
        new_train = []
        new_test = []
        new_val = []

        i = 0
        for data in self.data_pipe.image_clinical.X_train:
            for array in data:
                array = array[self.collected_indices_train]
                new_train.append(array)

                i = i + 1
        
        i = 0
        for data in self.data_pipe.image_clinical.X_test:
            for array in data:
                array = array[self.collected_indices_test]
                new_test.append(array)

                i = i + 1

        i = 0
        for data in self.data_pipe.image_clinical.X_val:
            for array in data:
                array = array[self.collected_indices_val]
                new_val.append(array)

                i = i + 1

        self.data_pipe.image_clinical.X_train = new_train
        self.data_pipe.image_clinical.X_test = new_test
        self.data_pipe.image_clinical.X_val = new_val

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
        