import os
from keras.backend import argmax
from keras.models import load_model
from scipy.sparse import data
from sklearn.metrics import accuracy_score
import numpy as np
from src.cancer_ml import cancer_ml

class voting_ensemble:

    # input iterables containing [x, y] for each data form
    def __init__(self):
        clinical = cancer_ml("metabric", "chemotherapy", model="clinical_only")
        image_clinical = cancer_ml("duke", "Adjuvant Chemotherapy", model="image_clinical")
        image_only = cancer_ml("duke", "Adjuvant Chemotherapy", model="cnn")

        clinical.run_model()
        clinical.test_model()

        image_clinical.run_model()
        image_clinical.test_model()

        image_only.run_model()
        image_only.test_model()

        self.clinical_models = self.load_models('data\\saved_models\\clinical')
        self.image_clinical_models = self.load_models('data\\saved_models\\image_clinical')
        self.image_only_models = self.load_models('data\\saved_models\\image_only')

        all_models = [self.clinical_models, self.image_clinical_models, self.image_only_models]

        clinical_test = [clinical.data_pipe.only_clinical.X_test, clinical.data_pipe.only_clinical.y_test]
        image_clinical_test = [image_clinical.data_pipe.image_clinical.X_test, image_clinical.data_pipe.image_clinical.y_test]
        image_only_test = [image_only.data_pipe.image_only.X_test, image_only.data_pipe.image_only.y_test]

        all_data = [clinical_test, image_clinical_test, image_only_test]

        i = 0
        predictions = []
        evals = []
        for models in all_models:

            # filter out empty model lists
            if len(models) >= 1:
                data = all_data[i]

                testX = data[0]
                testY = data[1]

                ensemble_prediction = self.predict(testX, models)
                predictions.append(ensemble_prediction)

                ensemble_eval = self.evaluate_models(testX, testY, models)
                evals.append(ensemble_eval)

            i = i + 1

        ensembled_predictions = []
        ensembled_evals = []
            
        # find average of predictions and evals for ensembled values
        for prediction in predictions:
            ensembled_predictions.append(float(np.mean(prediction)))

        for eval in evals:
            ensembled_evals.append(float(np.mean(eval)))

        self.ensembled_prediction = sum(ensembled_predictions) / len(ensembled_predictions)
        self.ensembled_eval = sum(ensembled_evals) / len(ensembled_evals)
        
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
