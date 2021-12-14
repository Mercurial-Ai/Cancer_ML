from tensorflow import keras
from tensorflow.keras.layers import Dense
from src.get_weight_dict import get_weight_dict
from src.grid_search.grid_search import grid_search
from src.confusion_matrix import confusion_matrix

class clinical_only:

    def __init__(self, load_model=True):
        self.load_model = load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):

        input = keras.layers.Input(shape=(X_train.shape[1],))

        x = Dense(9, activation='relu')(input)
        x = Dense(9, activation='relu')(x)
        x = Dense(6, activation='relu')(x)
        x = Dense(4, activation='relu')(x)
        x = Dense(2, activation='relu')(x)
        output = Dense(1, activation='linear')(x)
        self.model = keras.Model(input, output)

        #search = grid_search()
        #search.test_model(model, X_train, y_train, X_val, y_val, get_weight_dict(y_train))

        self.model.compile(optimizer='SGD',
                            loss='mean_squared_error',
                            metrics=['accuracy'])

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), class_weight=get_weight_dict(y_train))

        self.model.save('data\\saved_models\\clinical\\keras_clinical_only_model.h5')

        return self.model

    def test_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, batch_size=128)

        confusion_matrix(y_true=y_test, y_pred=self.model.predict(X_test))

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=32):
        
        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\keras_clinical_only_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
