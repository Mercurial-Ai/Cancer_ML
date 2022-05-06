from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import concatenate
from torch.nn.modules.loss import BCELoss
from src.class_loss import class_loss
from src.grid_search.grid_search import grid_search
from src.get_weight_dict import get_weight_dict
from src.confusion_matrix import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from src.metrics import recall_m, precision_m, f1_m
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
        if len(data) == 1:
            data = data[0]
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

    def train_func(self, X_train, y_train, epochs, batch_size, optimizer, criterion):
        for epoch in range(int(epochs)):
            running_loss = 0.0
            for i in range((X_train[0][1].shape[0]-1)//batch_size + 1):
                start_i = i*batch_size
                end_i = start_i+batch_size

                xb = [torch.from_numpy(X_train[0][0][start_i:end_i]), torch.from_numpy(X_train[0][1][start_i:end_i])]
                yb = torch.from_numpy(y_train[start_i:end_i]).type(torch.float)
                y_val = torch.from_numpy(y_train[start_i:end_i]).detach()

                xb[1] = torch.reshape(xb[1], (-1, 1, 256, 256))
                xb[1] = xb[1].type(torch.float)
                xb[0] = xb[0].type(torch.float)
                yb = yb.type(torch.float)
                y_val = y_val.type(torch.float)
                pred = self(xb)

                if type(criterion) == type(nn.CrossEntropyLoss()):
                    yb = yb.to(torch.long)

                # if variable is binary BCE should be used instead of Cross-Entropy
                if torch.unique(yb).shape[0] > 2:
                    loss = criterion(pred, yb)
                else:
                    yb = yb.unsqueeze(1).type(torch.float)
                    pred = torch.abs(torch.round(pred))
                    criterion = BCELoss()
                    loss = criterion(pred, yb)
        
                loss.backward()
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

                # print stats
                running_loss += loss.item()
                pred = pred.detach().numpy()
                y_val = y_val.detach().numpy().astype(np.float)
                pred = np.argmax(pred, axis=1).astype(np.float)
                self.loss = running_loss
                pred = pred.flatten()
                self.accuracy = accuracy_score(yb, pred)
                self.f1_score = f1_m(yb, pred)
                self.recall = recall_m(yb, pred)
                self.balanced_acc = balanced_accuracy_score(yb, pred)
                if i % 2000 == 1999: # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}', 'Accuracy: %.4f' %self.accuracy, 'F1: %.4f' %self.f1_score, 'Recall: %.4f' %self.recall, 'Balanced Accuracy: %.4f' %self.balanced_acc)
                running_loss = 0.0

        print("Finished Training")

        torch.save(self.state_dict(), "data/saved_models/image_clinical/torch_image_clinical_model.h5")

class image_model:

    def __init__(self, load_model=True):
        self.load_model = load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):

        if len(y_train.shape) > 1:
            self.multi_target = True
        else:
            self.multi_target = False

        self.model = image_clinical()

        if len(X_val) == 1:
            X_val = X_val[0]

        X_val = [torch.from_numpy(X_val[0]).type(torch.float), torch.from_numpy(X_val[1]).type(torch.float)]
        X_val[1] = torch.reshape(X_val[1], (-1, 1, 256, 256))

        search = grid_search()
        search.test_model(self.model, X_train, y_train, X_val, y_val, num_combs=5)

        self.criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.model.train_func(X_train, y_train, epochs, batch_size, optimizer, self.criterion)

        return self.model

    def test_model(self, X_test, y_test):
        X_test[0][1] = np.reshape(X_test[0][1], (X_test[0][1].shape[0], 1, 256, 256))
        X_test = [torch.from_numpy(X_test[0][0]).type(torch.float), torch.from_numpy(X_test[0][1]).type(torch.float)]
        y_test = torch.from_numpy(y_test).type(torch.float)
        with torch.no_grad():
            self.model.eval()
            y_pred = self.model(X_test)
            confusion_matrix(y_test, y_pred, save_name="image_only_c_mat_torch")
            test_loss = self.criterion(y_pred, y_test)

        return test_loss

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=128):
        
        if self.load_model:
            self.model = torch.load('data\\saved_models\\torch_image_clinical_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
        