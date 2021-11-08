from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from src.get_weight_dict import get_weight_dict

class cnn:

    def __init__(self, load_model=True):
        self.load_model=load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):

        opt = keras.optimizers.SGD(learning_rate=0.001)

        model = Sequential()

        model.add(layers.Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())

        model.add(layers.Dense(128))
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(64))
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(1))
        model.add(layers.Activation('linear'))

        model.compile(loss='mae',
                    optimizer=opt,
                    metrics=['accuracy'])

        self.fit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=get_weight_dict(y_train))

        model.save('data\\saved_models\\text_prediction\\keras_image_clinical_model.h5')

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
