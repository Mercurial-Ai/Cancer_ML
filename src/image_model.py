from tensorflow import keras
from tensorflow.keras.layers import Dense
from src.grid_search.grid_search import grid_search
from src.get_weight_dict import get_weight_dict

class image_model:

    def __init__(self, load_model=True):
        self.load_model = load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):
        input = keras.layers.Input(shape=(X_train.shape[1],))

        x = Dense(150000, activation="relu")(x)
        x = Dense(90000, activation="relu")(x)
        x = Dense(4500, activation="relu")(x)
        x = Dense(2000, activation='relu')(x)
        x = Dense(1000, activation='relu')(x)
        x = Dense(500, activation='relu')(x)
        x = Dense(200, activation='relu')(x)
        x = Dense(100, activation='relu')(x)
        x = Dense(50, activation='relu')(x)
        x = Dense(20, activation='relu')(x)
        output = Dense(1, activation='linear')(x)
        model = keras.Model(input, output)

        search = grid_search()
        search.test_model(model, X_train, y_train, X_val, y_val, get_weight_dict(y_train))

        model.compile(optimizer='sgd',
                            loss='mse',
                            metrics=['accuracy'])

        self.fit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), class_weight=get_weight_dict(y_train))

        model.save('data\\saved_models\\keras_image_clinical_model.h5')

        return model

    def test_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, batch_size=128)

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=128):
        
        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\keras_image_clinical_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
        