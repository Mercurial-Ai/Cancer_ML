import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import pandas as pd
from src.class_loss import class_loss
from src.confusion_matrix import confusion_matrix
from src.get_weight_dict import get_weight_dict
from src.grid_search.grid_search import grid_search
from src.metrics import recall_m, precision_m, f1_m

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

        self.res = tf.keras.applications.resnet50.ResNet50(input_shape=(256, 256, 1), include_top=False, weights=None)
        print(self.res.summary())

        self.model = keras.models.Sequential()
        self.model.add(self.res)
        self.model.add(keras.layers.Flatten())
        output = layers.Dense(1, activation='sigmoid')

        search = grid_search()

        if self.multi_target:
            search.test_model(self.model, X_train, y_train, X_val, y_val, num_combs=12)
        else:
            print("weights applied")
            search.test_model(self.model, X_train, y_train, X_val, y_val, get_weight_dict(y_train), num_combs=12)

        class_weights = get_weight_dict(y_train, output_names)

        if self.multi_target:
            self.model.compile(optimizer='adam',
                                loss={k: class_loss(v) for k, v, in class_weights.items()},
                                metrics=['accuracy', f1_m, precision_m, recall_m])

            self.fit = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        else:
            opt = keras.optimizers.Adam(lr=0.01)
            self.model.compile(loss='mse',
                    optimizer=opt,
                    metrics=['accuracy', f1_m, precision_m, recall_m])

            self.fit = self.model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_val, y_val), class_weight=get_weight_dict(y_train))

        try:
            self.model.save('data/saved_models/image_only/keras_cnn_model.h5')
        except:
            print("image only model could not be saved")

        return self.model

    def test_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, batch_size=32)

        confusion_matrix(y_true=y_test, y_pred=self.model.predict(X_test), save_name="image_only_c_mat.png")

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=32):

        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\keras_cnn_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
