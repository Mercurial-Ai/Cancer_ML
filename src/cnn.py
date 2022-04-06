from http.client import ResponseNotReady
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import pandas as pd
from src.class_loss import class_loss
from src.confusion_matrix import confusion_matrix
from src.get_weight_dict import get_weight_dict
from src.grid_search.grid_search import grid_search
from src.metrics import recall_m, precision_m, f1_m, BalancedSparseCategoricalAccuracy
from tensorflow.keras.metrics import AUC
from src.resnet18 import ResNet18

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

        # make black and white images have 3 channels
        X_train = np.squeeze(X_train)
        X_val = np.squeeze(X_val)
        X_train = np.stack((X_train,)*3, axis=-1)
        X_val = np.stack((X_val,)*3, axis=-1)

        self.res = ResNet18()
        self.res = self.res.build(input_shape=(256, 256, 3), num_classes=3)

        self.model = keras.models.Sequential()
        self.model.add(self.res)
        self.model.add(keras.layers.Flatten())
        self.model.add(layers.Dense(1, activation='sigmoid'))

        print(self.model.summary())

        search = grid_search()

        if self.multi_target:
            search.test_model(self.model, X_train, y_train, X_val, y_val, num_combs=12)
        else:
            print("weights applied")
            search.test_model(self.model, X_train, y_train, X_val, y_val, get_weight_dict(y_train), num_combs=12)

        opt = keras.optimizers.Adam(lr=0.01)
        auc_m = AUC()
        balanced_acc_m = BalancedSparseCategoricalAccuracy()
        self.model.compile(loss='mse',
                optimizer=opt,
                metrics=['accuracy', f1_m, precision_m, recall_m, auc_m, balanced_acc_m])

        self.fit = self.model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_val, y_val), class_weight=get_weight_dict(y_train))

        try:
            self.model.save('data/saved_models/image_only/keras_cnn_model.h5')
        except:
            print("image only model could not be saved")

        return self.model

    def test_model(self, X_test, y_test):

        X_test = np.squeeze(X_test)
        X_test = np.stack((X_test,)*3, axis=-1)
        
        results = self.model.evaluate(X_test, y_test, batch_size=32)

        confusion_matrix(y_true=y_test, y_pred=self.model.predict(X_test), save_name="image_only_c_mat.png")

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=32):

        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\keras_cnn_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
