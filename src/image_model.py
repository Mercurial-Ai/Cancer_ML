from tracemalloc import start
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import concatenate
from src.class_loss import class_loss
from src.grid_search.grid_search import grid_search
from src.get_weight_dict import get_weight_dict
from src.confusion_matrix import confusion_matrix
from src.metrics import recall_m, precision_m, f1_m, BalancedSparseCategoricalAccuracy
from tensorflow.keras.metrics import AUC
import torch
import torch.nn as nn
from src.resnet import resnet18
import numpy as np

class image_clinical(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.clinical_track()
        self.image_track()

    def clinical_track(self):
        self.fc1 = nn.Linear(603, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 15)

    def image_track(self):
        self.res = resnet18(pretrained=False)

    def forward(self, data):
        clinical_data = data[0]
        image_data = data[1]

        # clinical
        clinical_x = self.relu(self.fc1(clinical_data))
        clinical_x = self.relu(self.fc2(clinical_x))
        clinical_x = self.relu(self.fc3(clinical_x))

        # image
        image_x = self.res(image_data)

        x = torch.cat([clinical_x, image_x], dim=1)
        self.fc1_cat = nn.Linear(x.shape[-1], 26)
        x = self.relu(self.fc1_cat(x))
        self.fc2_cat = nn.Linear(26, 1)
        x = self.fc2_cat(x)

        return x

class image_model:

    def __init__(self, load_model=True):
        self.load_model = load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):

        if len(y_train.shape) > 1:
            self.multi_target = True
        else:
            self.multi_target = False

        self.model = image_clinical()

        self.criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epochs):
            running_loss = 0.0
            for i in range((X_train[0][1].shape[0]-1)//batch_size + 1):
                start_i = i*batch_size
                end_i = start_i+batch_size

                xb = [torch.from_numpy(X_train[0][0][start_i:end_i]), torch.from_numpy(X_train[0][1][start_i:end_i])]
                yb = torch.from_numpy(y_train[start_i:end_i])

                xb_image_shape = xb[1].shape

                xb[1] = torch.reshape(xb[1], (xb_image_shape[0], 1, 256, 256))
                xb[1] = xb[1].type(torch.float)
                xb[0] = xb[0].type(torch.float)
                pred = self.model(xb)
                pred = pred.to(torch.float)

                yb = torch.reshape(yb, (yb.shape[0], 1)).type(torch.float)
                loss = self.criterion(pred, yb)
        
                loss.backward()
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

                # print stats
                running_loss += loss.item()
                print('Completed training batch', epoch, 'Training Loss is: %.4f' %running_loss)
                running_loss = 0.0

        print("Finished Training")

        torch.save(self.model.state_dict(), "data/saved_models/image_clinical/torch_image_clinical_model.h5")

        return self.model

    def test_model(self, X_test, y_test):
        X_test[0][1] = np.reshape(X_test[0][1], (X_test[0][1].shape[0], 1, 256, 256))
        X_test = [torch.from_numpy(X_test[0][0]).type(torch.float), torch.from_numpy(X_test[0][1]).type(torch.float)]
        y_test = torch.from_numpy(y_test).type(torch.float)
        with torch.no_grad():
            self.model.eval()
            y_pred = self.model(X_test)
            test_loss = self.criterion(y_pred, y_test)

        return test_loss

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=128):
        
        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\keras_image_clinical_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
        