from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.core import Activation
from src.grid_search.grid_search import grid_search

class u_net:

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):

        def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
            c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
            c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
            p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
            return c, p

        def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
            us = keras.layers.UpSampling2D((2, 2))(x)
            concat = keras.layers.Concatenate()([us, skip])
            c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
            c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
            return c

        def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
            c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
            c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
            return c

        f = [16, 32, 64, 128, 256]
        inputs = keras.layers.Input((X_train.shape[1], X_train.shape[2], 1))
        
        p0 = inputs
        c1, p1 = down_block(p0, f[4]) 
        c2, p2 = down_block(p1, f[3]) 
        c3, p3 = down_block(p2, f[2]) 
        c4, p4 = down_block(p3, f[1]) 
        
        bn = bottleneck(p4, f[0])
        
        u1 = up_block(bn, c4, f[1]) 
        u2 = up_block(u1, c3, f[2]) 
        u3 = up_block(u2, c2, f[3]) 
        u4 = up_block(u3, c1, f[4]) 
        
        x = keras.layers.Conv2D(1, (1, 1), padding="same", activation="relu")(u4)
        x = keras.layers.MaxPooling2D((3, 3))(x)
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(y_train.shape[-1], activation='linear')(x)
        self.model = keras.models.Model(inputs, outputs)

        print(self.model.summary())

        search = grid_search()

        search.test_model(self.model, X_train, y_train, X_val, y_val, num_combs=12)

        self.model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

        return self.model

    def test_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, batch_size=32)

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=32):
        self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
