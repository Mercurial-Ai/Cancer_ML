from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
import tensorflow as tf

class weighted_loss(CategoricalCrossentropy):

    """

    var - column index that weight is applied to

    """
    
    def __init__(self, var, class_, weight):
        self.var = var
        self.class_ = class_
        self.weight = weight

    def loss_func(self, y_true, y_pred):

        i = 0
        losses = []
        for true in y_true:
            pred = y_pred[i]

            true_var = true[self.var]

            true = np.expand_dims(true, 0)
            pred = np.expand_dims(pred, 0)

            cc = CategoricalCrossentropy()
            loss = cc(true, pred).numpy()

            if self.class_ == true_var:
                loss = loss*self.weight

            losses.append(loss)

            i = i + 1

        loss = sum(losses) / len(losses)

        loss = round(loss, 2)

        loss = tf.convert_to_tensor(loss)

        return loss
