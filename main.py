from src.PeakCluster import PeakCluster
from src.data_pipeline import data_pipeline
from src.clinical_only import clinical_only
from src.image_model import image_model
from src.cnn import cnn

import numpy as np

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
        self.data_pipe = data_pipeline("data\\Duke-Breast-Cancer-MRI\\Clinical and Other Features (edited).csv", "data\Duke-Breast-Cancer-MRI\img_array_duke.npy", self.target)
        self.data_pipe.load_data()
    
    def collect_HN1(self):
        self.data_pipe = data_pipeline("data\\HNSCC-HN1\\Copy of HEAD-NECK-RADIOMICS-HN1 Clinical data updated July 2020 (adjusted chemotherapy var).csv", "data\\HNSCC-HN1\\img_array.npy", self.target)
        self.data_pipe.load_data()

    def collect_METABRIC(self):
        self.data_pipe = data_pipeline("data\\METABRIC_RNA_Mutation\\METABRIC_RNA_Mutation.csv", None, self.target)
        self.data_pipe.load_data()

    def run_model(self):

        if self.clinical:
            self.model = clinical_only(load_model=False)
            self.model = self.model.get_model(self.data_pipe.only_clinical.X_train, self.data_pipe.only_clinical.y_train, self.data_pipe.only_clinical.X_val, self.data_pipe.only_clinical.y_val)
        elif self.image_clinical:
            self.model = image_model(load_model=False)
            self.model = self.model.get_model(self.data_pipe.image_clinical.X_train, self.data_pipe.image_clinical.y_train, self.data_pipe.image_clinical.X_val, self.data_pipe.image_clinical.y_val)
        elif self.cnn:
            self.model = cnn(load_model=False)
            self.model = self.model.get_model(self.data_pipe.image_only.X_train, self.data_pipe.image_only.y_train, self.data_pipe.image_only.X_val, self.data_pipe.image_only.y_val)

    def make_class_inference(self, image_array):

        X = self.data_pipe.image_only.X_train

        X = np.reshape(X, (450, 262144))
        model = PeakCluster(X)

        image_array = image_array.flatten()
        image_array = np.expand_dims(image_array, axis=0)

        inference = model.predict(image_array)

        return inference

ml = cancer_ml('duke', 'Adjuvant Chemotherapy', model='cnn')
#ml.run_model()

array = ml.data_pipe.image_only.X_val

image1 = array[0]
image2 = array[1]
image3 = array[2]

print(ml.make_class_inference(image1))
print(ml.make_class_inference(image2))
print(ml.make_class_inference(image3))
