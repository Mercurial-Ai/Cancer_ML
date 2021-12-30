from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import concatenate
from src.grid_search.grid_search import grid_search
from src.get_weight_dict import get_weight_dict
from src.confusion_matrix import confusion_matrix
import numpy as np
import math
import pandas as pd

class image_model:

    def __init__(self, load_model=True):
        self.load_model = load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):

        if type(y_train) == pd.DataFrame:
            self.multi_target = True
        else:
            self.multi_target = False

        clinical_input = keras.layers.Input(shape=(X_train[0][0].shape[1]))

        x = Dense(50, activation="relu")(clinical_input)
        x = Dense(25, activation='relu')(x)
        x = Dense(15, activation='relu')(x)
        flat1 = keras.layers.Flatten()(x)

        image_input = keras.layers.Input(shape=(512, 512, 1))

        x = Conv2D(64, kernel_size=5, activation='relu')(image_input)
        x = MaxPooling2D(pool_size=(5, 5))(x)
        x = Conv2D(32, kernel_size=5, activation='relu')(x)
        x = MaxPooling2D(pool_size=(5, 5))(x)
        x = Conv2D(8, kernel_size=5, activation='relu')(x)
        x = MaxPooling2D(pool_size=(5, 5))(x)
        flat2 = keras.layers.Flatten()(x)

        merge = concatenate([flat1, flat2])

        x = Dense(80, activation='relu')(merge)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)

        output = Dense(y_train.shape[1], activation='linear')(x)
        model = keras.Model([clinical_input, image_input], output)

        print(model.summary())

        search = grid_search()

        if self.multi_target:
            search.test_model(model, X_train, y_train, X_val, y_val, num_combs=24)
        else:
            search.test_model(model, X_train, y_train, X_val, y_val, get_weight_dict(y_train), num_combs=24)

        model.compile(optimizer='sgd',
                            loss='mse',
                            metrics=['accuracy'])

        if self.multi_target:
            self.fit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        else:
            self.fit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), class_weight=get_weight_dict(y_train))

        try:
            model.save('data/saved_models/image_clinical/keras_image_clinical_model.h5')
        except:
            print("image clinical model could not be saved")

        return model

    def test_model(self, X_test, y_test):

        results = self.model.evaluate(X_test, y_test, batch_size=128)

        if len(y_test.shape) == 1:
            confusion_matrix(y_true=y_test, y_pred=self.model.predict(X_test))

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=128):
        
        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\keras_image_clinical_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
        