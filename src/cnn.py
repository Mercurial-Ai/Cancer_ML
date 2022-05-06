from tracemalloc import start
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import pandas as pd
import numpy as np
from src.class_loss import class_loss
from src.confusion_matrix import confusion_matrix
from src.get_weight_dict import get_weight_dict
from src.grid_search.grid_search import grid_search
from src.metrics import recall_m, precision_m, f1_m, BalancedSparseCategoricalAccuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class torch_cnn(nn.Module):
    def __init__(self):
        super(torch_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
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

    def train_func(self, X_train, y_train, epochs, batch_size, optimizer, criterion):
        for epoch in range(int(epochs)):
            running_loss = 0.0
            for i in range((X_train.shape[0]-1)//batch_size + 1):
                start_i = i*batch_size
                end_i = start_i+batch_size

                xb = X_train[start_i:end_i]
                yb = y_train[start_i:end_i]

                xb_shape = xb.shape

                xb = torch.from_numpy(xb)
                yb = torch.from_numpy(yb)

                xb = xb.to(device)
                yb = yb.to(device)

                xb = torch.reshape(xb, (xb_shape[0], 1, 256, 256))
                xb = xb.type(torch.float)
                pred = self(xb)

                pred = pred.to(torch.float)
                yb = yb.to(torch.long)

                try:
                    loss = criterion(pred, yb)
                except UserWarning as e:
                    pass
        
                loss.backward()
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

                # print stats
                running_loss += loss.item()
                pred = pred.detach().numpy()
                y_val = y_val.astype(np.float)
                pred = np.argmax(pred, axis=1).astype(np.float)
                pred = pred.flatten()
                self.loss = running_loss
                self.accuracy = accuracy_score(yb, pred)
                self.f1_score = f1_m(yb, pred)
                self.recall = recall_m(yb, pred)
                self.balanced_acc = balanced_accuracy_score(yb, pred)
                print('Completed training batch', epoch, 'Training Loss is: %.4f' %running_loss, 'Accuracy: %.4f' %self.accuracy, 'F1: %.4f' %self.f1_score, 'Recall: %.4f' %self.recall, 'Balanced Accuracy: %.4f' %self.balanced_acc)
                running_loss = 0.0

        print("Finished Training")

        torch.save(self.state_dict(), "data/saved_models/image_only/torch_cnn_model.h5")

class cnn:

    def __init__(self, load_model=True):
        self.load_model=load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):

        if len(y_train.shape) > 1:
            self.multi_target = True
        else:
            self.multi_target = False

        self.model = torch_cnn()
        self.model.to(device)

        search = grid_search()

        search.test_model(self.model, X_train, y_train, X_val, y_val, num_combs=5)

        self.criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.model.train_func(X_train, y_train, epochs, batch_size, optimizer, self.criterion)

        return self.model

    def test_model(self, X_test, y_test):
        X_test = np.reshape(X_test, (X_test.shape[0], 1, 256, 256))
        X_test = torch.from_numpy(X_test).type(torch.float)
        y_test = torch.from_numpy(y_test).type(torch.long)
        with torch.no_grad():
            self.model.eval()
            y_pred = self.model(X_test)

            confusion_matrix(y_test, y_pred, save_name="image_only_c_mat_torch")
            test_loss = self.criterion(y_pred, y_test)

        return test_loss

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=32):

        if self.load_model:
            self.model = torch.load('data\\saved_models\\torch_cnn_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
