from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import concatenate
from src.class_loss import class_loss
from src.grid_search.grid_search import grid_search
from src.get_weight_dict import get_weight_dict
from src.confusion_matrix import confusion_matrix
from src.metrics import recall_m, precision_m, f1_m
from tensorflow.keras.metrics import AUC

class image_model:

    def __init__(self, load_model=True):
        self.load_model = load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):

        if len(y_train.shape) > 1:
            self.multi_target = True
        else:
            self.multi_target = False

        print("X train shape clinical:", X_train[0][0].shape)

        clinical_input = keras.layers.Input(shape=(X_train[0][0].shape[1]))

        x = Dense(50, activation="relu")(clinical_input)
        x = Dense(25, activation='relu')(x)
        x = Dense(15, activation='relu')(x)
        flat1 = keras.layers.Flatten()(x)

        image_input = keras.layers.Input(shape=X_train[0][1].shape[1:])

        x = Conv2D(64, kernel_size=4, activation='relu')(image_input)
        x = MaxPooling2D(pool_size=(4, 4))(x)
        x = Conv2D(32, kernel_size=5, activation='relu')(x)
        x = MaxPooling2D(pool_size=(4, 5))(x)
        x = Conv2D(8, kernel_size=5, activation='relu')(x)
        x = MaxPooling2D(pool_size=(4, 4))(x)
        flat2 = keras.layers.Flatten()(x)

        merge = concatenate([flat1, flat2])

        x = Dense(26, activation='relu')(merge)

        if self.multi_target:
            outputs = []
            for i in range(y_train.shape[-1]):
                output = Dense(1, activation='sigmoid')(x)

                outputs.append(output)
        else:
            outputs = Dense(1, activation='sigmoid')(x)

        model = keras.Model([clinical_input, image_input], outputs)

        print(model.summary())

        output_names = []
        for layer in model.layers:
            if type(layer) == Dense:
                if layer.units == 1:
                    output_names.append(layer.name)

        search = grid_search()

        if self.multi_target:
            search.test_model(model, X_train, y_train, X_val, y_val, num_combs=12)
        else:
            search.test_model(model, X_train, y_train, X_val, y_val, get_weight_dict(y_train), num_combs=12)

        class_weights = get_weight_dict(y_train, output_names)

        auc_m = AUC()
        if self.multi_target:

            model.compile(optimizer='adam',
                            loss={k: class_loss(v) for k, v, in class_weights.items()},
                            metrics=['accuracy', f1_m, precision_m, recall_m, auc_m])

            self.fit = model.fit(X_train[0], y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val))
        else:
            model.compile(optimizer='adam',
                                    loss='mae',
                                    metrics=['accuracy', f1_m, precision_m, recall_m, auc_m])

            self.fit = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val), class_weight=class_weights)

        try:
            model.save('data/saved_models/image_clinical/keras_image_clinical_model.h5')
        except:
            print("image clinical model could not be saved")

        return model

    def test_model(self, X_test, y_test):

        results = self.model.evaluate(X_test, y_test, batch_size=32)

        confusion_matrix(y_true=y_test, y_pred=self.model.predict(X_test), save_name="image_clinical_c_mat.png")

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=128):
        
        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\keras_image_clinical_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
        