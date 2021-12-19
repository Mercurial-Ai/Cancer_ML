from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from src.confusion_matrix import confusion_matrix
from src.get_weight_dict import get_weight_dict

from src.grid_search.grid_search import grid_search

class cnn:

    def __init__(self, load_model=True):
        self.load_model=load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=128):

        opt = keras.optimizers.SGD(learning_rate=0.007)
        loss = keras.losses.BinaryCrossentropy()

        self.model = Sequential()

        self.model.add(layers.Conv2D(64, (6, 6), input_shape=X_train.shape[1:]))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(4, 4)))

        self.model.add(layers.Conv2D(32, (6, 6)))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3)))

        self.model.add(layers.Conv2D(16, (6, 6)))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(16))
        self.model.add(layers.Activation('relu'))

        self.model.add(layers.Dense(8))
        self.model.add(layers.Activation('relu'))

        self.model.add(layers.Dense(1))
        self.model.add(layers.Activation('linear'))

        search = grid_search()
        search.test_model(self.model, X_train, y_train, X_val, y_val, get_weight_dict(y_train))

        self.model.compile(loss=loss,
                    optimizer=opt,
                    metrics=['accuracy'])

        self.fit = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), class_weight=get_weight_dict(y_train))

        self.model.save('data/saved_models/image_only/keras_cnn_model.h5')

        return self.model

    def test_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, batch_size=128)

        confusion_matrix(y_true=y_test, y_pred=self.model.predict(X_test))

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=32):

        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\keras_cnn_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
