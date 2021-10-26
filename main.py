from src.PeakCluster import PeakCluster
from src.data_pipeline import data_pipeline
from src.clinical_only import clinical_only
from src.image_model import image_model
from src.cnn import cnn
from src.get_distribution import get_distribution

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

class cancer_ml:

    def __init__(self, dataset, target, model="clinical_only"):
        self.dataset = dataset
        self.target = target

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
            model = self.model.get_model(self.data_pipe.only_clinical.X_train, self.data_pipe.only_clinical.y_train, self.data_pipe.only_clinical.X_val, self.data_pipe.only_clinical.y_val)
        elif self.image_clinical:
            self.model = image_model(load_model=False)
            model = self.model.get_model(self.data_pipe.image_clinical.X_train, self.data_pipe.image_clinical.y_train, self.data_pipe.image_clinical.X_val, self.data_pipe.image_clinical.y_val)
        elif self.cnn:
            self.model = cnn(load_model=False)
            model = self.model.get_model(self.data_pipe.image_only.X_train, self.data_pipe.image_only.y_train, self.data_pipe.image_only.X_val, self.data_pipe.image_only.y_val)

    def test_model(self):
        if self.clinical:
            print(self.model.test_model(self.data_pipe.only_clinical.X_test, self.data_pipe.only_clinical.y_test))
        elif self.image_clinical:
            print(self.model.test_model(self.data_pipe.image_clinical.X_test, self.data_pipe.image_clinical.y_test))
        elif self.cnn:
            print(self.model.test_model(self.data_pipe.image_only.X_test, self.data_pipe.image_only.y_test))

    def setup_cluster(self):
        X = self.data_pipe.image_only.X_train

        X = np.reshape(X, (450, 262144))
        self.model = PeakCluster(X)

    def make_class_inference(self, image_array):

        image_array = image_array.flatten()
        image_array = np.expand_dims(image_array, axis=0)

        inference = self.model.predict(image_array)

        return inference

    def k_neighbors(self):

        X = self.data_pipe.image_only.X_train

        X = np.reshape(X, (450, 262144))

        print(len(self.model.labels_))

        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X, self.model.labels_)
        
        print(get_distribution(self.model.labels_))

        # identify label index with class 2
        i = 0
        for label in self.model.labels_:
            
            if label == 2:
                class_index = i

            i = i + 1

        example_image = X[class_index]

        example_image = np.reshape(example_image, (1, -1))

        prediction = neigh.predict(example_image)[0]

        print(prediction)

    def get_classes(self):
        return self.model.labels_

ml = cancer_ml('duke', 'Adjuvant Chemotherapy', model='cnn')
#ml.run_model()
#ml.test_model()

ml.setup_cluster()
print(ml.get_classes())

ml.k_neighbors()
