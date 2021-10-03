from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanIoU

class image_model:

    def __init__(self, load_model=True):
        self.load_model = load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        input = keras.layers.Input(shape=(self.X_train.shape[1],))

        x = Dense(150, activation=self.activation_function)(input)
        x = Dense(150, activation=self.activation_function)(x)
        x = Dense(150, activation=self.activation_function)(x)
        x = Dense(120, activation=self.activation_function)(x)
        x = Dense(120, activation=self.activation_function)(x)
        x = Dense(100, activation=self.activation_function)(x)
        x = Dense(100, activation=self.activation_function)(x)
        x = Dense(80, activation=self.activation_function)(x)
        x = Dense(80, activation=self.activation_function)(x)
        x = Dense(45, activation=self.activation_function)(x)
        output = Dense(1, activation='linear')(x)
        model = keras.Model(input, output)

        model.compile(optimizer='sgd',
                            loss='mse',
                            metrics=['accuracy'])

        self.fit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

        model.save('data\\saved_models\\text_prediction\\keras_image_clinical_model.h5')

        return model

    def test_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, batch_size=128)

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=None, batch_size=None):
        
        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\text_prediction\\keras_image_clinical_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
        