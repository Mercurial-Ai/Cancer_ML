import os
from keras.backend import argmax
from keras.models import load_model
from sklearn.metrics import accuracy_score
import numpy as np

class voting_ensemble:

    # input iterables containing [x, y] for each data form
    def __init__(self, clinical_test, image_clinical_test, image_only_test):
        self.clinical_test = clinical_test
        self.image_clinical_test = image_clinical_test
        self.image_only_test = image_only_test

        self.clinical_models = self.load_models('data\\saved_models\\clinical')
        self.image_clinical_models = self.load_models('data\\saved_models\\image_clinical')
        self.image_only_models = self.load_models('data\\saved_models\\image_only')

        print(self.predict(self.image_only_test[0], self.image_only_models))
        
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

    def evaluate_models(self, testX, testY):

        y = self.predict(testX)

        return accuracy_score(testY, y)
