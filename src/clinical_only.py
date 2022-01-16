from matplotlib.pyplot import get
from tensorflow import keras
from tensorflow.keras.layers import Dense
from src.get_weight_dict import get_weight_dict
from src.grid_search.grid_search import grid_search
from src.confusion_matrix import confusion_matrix
from src.weighted_loss import weighted_loss
import pandas as pd

class clinical_only:

    def __init__(self, load_model=True):
        self.load_model = load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):

        if len(y_train.shape) > 1:
            self.multi_target = True
        else:
            self.multi_target = False

        # use shape of data to determine which dataset is being utilized (METABRIC or Duke)
        if X_train.shape != (1713, 691):

            input = keras.layers.Input(shape=(X_train.shape[1],))

            x = Dense(64, activation='relu')(input)
            x = Dense(32, activation='relu')(x)
            output = Dense(y_train.shape[-1], activation='linear')(x)

            self.model = keras.Model(input, output)

            print(self.model.summary())

            search = grid_search()

            if self.multi_target:
                search.test_model(self.model, X_train, y_train, X_val, y_val, num_combs=1)
            else:
                search.test_model(self.model, X_train, y_train, X_val, y_val, get_weight_dict(y_train), num_combs=1)

            self.model.compile(optimizer='SGD',
                                loss='mean_squared_error',
                                metrics=['accuracy'])

            if self.multi_target:
                self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
            else:
                self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), class_weights=get_weight_dict(y_train))

        else:
   
            input = keras.layers.Input(shape=(X_train.shape[1],))

            x = Dense(512, activation='relu')(input)
            x = Dense(256, activation='relu')(x)
            x = Dense(128, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            x = Dense(32, activation='relu')(x)
            x = Dense(16, activation='relu')(x)

            output = Dense(y_train.shape[-1], activation='linear')(x)

            self.model = keras.Model(input, output)

            print(self.model.summary())

            search = grid_search()

            if self.multi_target:
                search.test_model(self.model, X_train, y_train, X_val, y_val, num_combs=24)
            else:
                search.test_model(self.model, X_train, y_train, X_val, y_val, get_weight_dict(y_train), num_combs=24)

            loss = weighted_loss(5, 0, 1.5)

            print("loss applied")
            self.model.compile(optimizer='SGD',
                                loss=loss.loss_func,
                                metrics=['accuracy'])

            if self.multi_target:
                self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
            else:
                self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), class_weights=get_weight_dict(y_train))

        # use shape of data to determine which dataset is being utilized (METABRIC or Duke)
        if X_train.shape == (1713, 691):
            try:
                self.model.save('data/saved_models/clinical_metabric/keras_clinical_only_model.h5')
            except:
                print("clinical metabric could not be saved")
        else:
            try:
                self.model.save('data/saved_models/clinical_duke/keras_clinical_only_model.h5')
            except:
                print("clinical duke could not be saved")

        return self.model

    def test_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, batch_size=32)

        if len(y_test.shape) == 1:
            try:
                confusion_matrix(y_true=y_test, y_pred=self.model.predict(X_test))
            except:
                print('c matrix failed')

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=32):
        
        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\keras_clinical_only_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
