from tracemalloc import start
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import pandas as pd
from src.class_loss import class_loss
from src.confusion_matrix import confusion_matrix
from src.get_weight_dict import get_weight_dict
from src.grid_search.grid_search import grid_search
from src.metrics import recall_m, precision_m, f1_m, BalancedSparseCategoricalAccuracy
import torch
import torch.nn as nn
import torch.nn.functional as F

class torch_cnn(nn.Module):
    def __init__(self):
        super(torch_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, device='cpu', dtype=torch.float)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

class cnn:

    def __init__(self, load_model=True):
        self.load_model=load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):

        if len(y_train.shape) > 1:
            self.multi_target = True
        else:
            self.multi_target = False

        self.model = torch_cnn()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epochs):
            running_loss = 0.0
            for i in range((X_train.shape[0]-1)//batch_size + 1):
                start_i = i*batch_size
                end_i = start_i+batch_size

                print(X_train.shape)

                xb = X_train[start_i:end_i]
                yb = y_train[start_i:end_i]

                xb_shape = xb.shape

                xb = torch.from_numpy(xb)
                yb = torch.from_numpy(yb)

                xb = torch.reshape(xb, (xb_shape[0], 1, 256, 256))
                xb = xb.type(torch.float)
                pred = self.model(xb)

                pred = pred.to(torch.float)
                yb = yb.to(torch.long)
                loss = criterion(pred, yb)
        
                loss.backward()
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

                # print stats
                running_loss += loss.item()
                if i % 2000 == 1999: # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print("Finished Training")

        torch.save(self.model.state_dict(), "data/saved_models/image_only/torch_cnn_model.h5")

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
