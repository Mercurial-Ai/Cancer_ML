from src.PeakCluster import PeakCluster
from src.data_pipeline import data_pipeline
from src.clinical_only import clinical_only
from src.image_model import image_model
from src.cnn import cnn
from collections import Counter
import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from src.random_crop import random_crop

class cancer_ml:

    def __init__(self, dataset, target, model="clinical_only", crop_size=(256, 256)):
        self.dataset = dataset
        self.target = target
        self.crop_size=crop_size

        # initialize model bools
        self.clinical = False
        self.image_clinical = False
        self.cnn = False

        if model == "clinical_only":
            self.clinical = True
        elif model == "image_clinical":
            self.image_clinical = True
        elif model == "cnn":
            self.cnn = True

        if self.dataset == "duke":
            self.collect_duke()
        elif self.dataset == "hn1":
            self.collect_HN1()
        elif self.dataset == "metabric":
            self.collect_METABRIC()

    def collect_duke(self):
        self.data_pipe = data_pipeline("data/Duke-Breast-Cancer-MRI/Clinical and Other Features (edited).csv", "data/Duke-Breast-Cancer-MRI/img_array_duke.npy", self.target)
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

        X = random_crop(X, (self.crop_size[0], self.crop_size[1], 1))

        # flatten X for KNeighbors
        X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2])).astype('float32')

        self.neigh = KNeighborsClassifier(n_neighbors=3)
        self.neigh.fit(X, self.model.labels_)

        # unflatten X
        X = np.reshape(X, (-1, int(math.sqrt(X.shape[1])), int(math.sqrt(X.shape[1]))))
        print(X.shape)

        self.data_pipe.image_only.X_train = self.equalize_classes(X)

    def get_classes(self):
        return self.model.labels_

    def divide_into_classes(self, image_array, n_clusters):

        y_train = self.data_pipe.image_only.y_train

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

                y = y_train[collected_indices]

                class_y_dict[label] = y

                self.num_clusters_used = self.num_clusters_used + 1

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
        
        for label in list(class_y_dict.keys()):

            class_array = class_y_dict[label]
            new_y = class_array[:lowest_count]
            class_y_dict[label] = new_y

        new_y = np.array([])
        for array in list(class_y_dict.values()):
            new_y = np.append(new_y, array)

        self.data_pipe.image_only.y_train = new_y

        for label in list(class_array_dict.keys()):

            class_array = class_array_dict[label]
            new_array = class_array[:lowest_count]
            class_array_dict[label] = new_array

        print(lowest_count*self.num_clusters_used)
        new_data = np.empty(shape=(lowest_count*self.num_clusters_used, self.crop_size[0], self.crop_size[1]), dtype=np.int8)
        i = 0
        for image_array in list(class_array_dict.values()):

            for image in image_array:
                new_data[i] = image

                i = i + 1

        new_data = np.expand_dims(new_data, axis=-1)

        return new_data

ml = cancer_ml('duke', 'Adjuvant Chemotherapy', model='cnn')

ml.setup_cluster()
ml.k_neighbors()

ml.run_model()
