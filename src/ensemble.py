import os
from keras.backend import argmax
from keras.models import load_model
from scipy.sparse import data
from sklearn.metrics import accuracy_score
import numpy as np

class data_pod:
    def __init__(self, models, data):
        self.models = models
        self.data = data

class voting_ensemble:

    # input iterables containing [x, y] for each data form
    def __init__(self, clinical_test, image_clinical_test, image_only_test):
        self.clinical_test = clinical_test
        self.image_clinical_test = image_clinical_test
        self.image_only_test = image_only_test

        self.clinical_models = self.load_models('data\\saved_models\\clinical')
        self.image_clinical_models = self.load_models('data\\saved_models\\image_clinical')
        self.image_only_models = self.load_models('data\\saved_models\\image_only')

        clinical_only = data_pod(self.clinical_models, self.clinical_test)
        image_clincial = data_pod(self.image_clinical_models, image_clinical_test)
        image_only = data_pod(self.image_only_models, self.image_only_test)

        all_models = [clinical_only, image_clincial, image_only]

        for pod in all_models:
            testX = pod.data[0]
            testY = pod.data[1]
            models = pod.models 
           
            try:
                print('Ensemble Prediction:', self.predict(testX, models))
                print('Ensemble Eval:', self.evaluate_models(testX, testY, models))
            except IndexError:
                print("Warning: empty model list")
        
    def load_models(self, model_dir):

        model_names = os.listdir(model_dir)

        model_paths = list()
        for name in model_names:
            full_path = os.path.join(model_dir, name)
            model_paths.append(full_path)

        models = list()
        for path in model_paths:
            model = load_model(path)
            models.append(model)

        return models

    def predict(self, testX, models):

        y = [model.predict(testX) for model in models]
        y = np.array(y)

        sum = np.sum(y, axis=0)

        result = argmax(sum, axis=1)

        return result

    def evaluate_models(self, testX, testY, models):

        y = self.predict(testX, models)

        return accuracy_score(testY, y)
