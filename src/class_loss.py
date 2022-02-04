import tensorflow as tf

def class_loss(class_weight):

    class_weight = tf.convert_to_tensor(class_weight)

    def loss(y_true, y_pred):
        y_true = tf.dtypes.cast(y_true, tf.int32)
        hothot = tf.one_hot(tf.reshape(y_true, [-1]), depth=class_weight.shape[0])
        weight = tf.math.multiply(class_weight, hothot)
        weight = tf.reduce_sum(weight, axis=-1)
        losses = tf.keras.losses.mean_absolute_error(y_true, y_pred)

        return losses

    return loss
