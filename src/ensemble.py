import os
from keras.backend import argmax
from keras.models import load_model
from sklearn.metrics import accuracy_score
import numpy as np

class voting_ensemble:
    def load_models(self, model_dir):

        model_names = os.listdir(model_dir)

        model_paths = list()
        for name in model_names:
            full_path = os.path.join(model_dir, name)
            model_paths.append(full_path)

        self.models = list()
        for path in model_paths:
            model = load_model(path)
            self.models.append(model)

    def predict(self, testX):
        y = [model.predict(testX) for model in self.models]
        y = np.array(y)

        sum = np.sum(y, axis=0)

        result = argmax(sum, axis=1)

        return result

    def evaluate_models(self, testX, testY):
        y = self.predict(testX)

        return accuracy_score(testY, y)
