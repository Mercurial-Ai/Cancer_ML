from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from src.get_weight_dict import get_weight_dict

from src.grid_search.grid_search import grid_search

class cnn:

    def __init__(self, load_model=True):
        self.load_model=load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=128):

        opt = keras.optimizers.SGD(learning_rate=0.007)
        loss = keras.losses.BinaryCrossentropy()

        model = Sequential()

        model.add(layers.Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(32, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(16, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())

        model.add(layers.Dense(16))
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(8))
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(4))
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(1))
        model.add(layers.Activation('linear'))

        search = grid_search()
        search.test_model(model, X_train, y_train, X_val, y_val, get_weight_dict(y_train))

        model.compile(loss=loss,
                    optimizer=opt,
                    metrics=['accuracy'])

        self.fit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), class_weight=get_weight_dict(y_train))

        model.save('data\\saved_models\\text_prediction\\keras_cnn_model.h5')

        return model

    def test_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, batch_size=128)

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=32):

        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\text_prediction\\keras_cnn_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
