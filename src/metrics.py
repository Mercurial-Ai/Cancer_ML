import torch
import keras

def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true*y_pred, 0, 1).type(torch.float)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1).type(torch.float)))
    recall = true_positives / (possible_positives + torch.finfo(torch.float32).eps)
    return recall

def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true*y_pred, 0, 1).type(torch.float)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1).type(torch.float)))
    precision = true_positives / (predicted_positives + torch.finfo(torch.float32).eps)
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+torch.finfo(torch.float32).eps))

class BalancedSparseCategoricalAccuracy(keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='balanced_sparse_categorical_accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = torch.squeeze(y_flat, axis=[-1])
        y_true_int = torch.cast(y_flat, torch.int32)

        cls_counts = torch.math.bincount(y_true_int)
        cls_counts = torch.math.reciprocal_no_nan(torch.cast(cls_counts, self.dtype))
        weight = torch.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)
