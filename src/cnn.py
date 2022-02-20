from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import pandas as pd
from src.class_loss import class_loss
from src.confusion_matrix import confusion_matrix
from src.get_weight_dict import get_weight_dict

from src.grid_search.grid_search import grid_search

class cnn:

    def __init__(self, load_model=True):
        self.load_model=load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=128):

        if len(y_train.shape) > 1:
            self.multi_target = True
        else:
            self.multi_target = False

        opt = keras.optimizers.SGD(learning_rate=0.007)
        loss = keras.losses.BinaryCrossentropy()

        input = layers.Input(shape=(X_train.shape[1:]))

        x = layers.Conv2D(64, (4, 4))(input)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(4, 4))(x)

        x = layers.Conv2D(32, (4, 4))(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)

        x = layers.Conv2D(16, (4, 4))(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Flatten()(x)

        x = layers.Dense(512)(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(256)(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(128)(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(64)(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(32)(x)
        x = layers.Activation('relu')(x)

        if self.multi_target:
            outputs = []
            for i in range(y_train.shape[-1]):
                output = layers.Dense(1, activation='sigmoid')(x)

                outputs.append(output)
        else:
            outputs = layers.Dense(1, activation='sigmoid')(x)

        self.model = keras.Model(input, outputs)

        output_names = []
        for layer in self.model.layers:
            if type(layer) == layers.Dense:
                if layer.units == 1:
                    output_names.append(layer.name)

        search = grid_search()

        #  if self.multi_target:
        #     search.test_model(self.model, X_train, y_train, X_val, y_val, num_combs=12)
        #else:
        #   print("weights applied")
        #  search.test_model(self.model, X_train, y_train, X_val, y_val, get_weight_dict(y_train), num_combs=12)

        class_weights = get_weight_dict(y_train, output_names)

        if self.multi_target:
            self.model.compile(optimizer='adam',
                                loss="mse",
                                metrics=['accuracy'])

            self.fit = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        else:
            self.model.compile(loss=loss,
                    optimizer=opt,
                    metrics=['accuracy'])

            self.fit = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), class_weight=get_weight_dict(y_train))

        try:
            self.model.save('data/saved_models/image_only/keras_cnn_model.h5')
        except:
            print("image only model could not be saved")

        return self.model

    def test_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, batch_size=32)

        print(self.model.predict(X_test))

        if len(y_test.shape) == 1:
            confusion_matrix(y_true=y_test, y_pred=self.model.predict(X_test))

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=32):

        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\keras_cnn_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
