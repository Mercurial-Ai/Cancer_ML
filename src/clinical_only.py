from matplotlib.pyplot import get
from tensorflow import keras
from tensorflow.keras.layers import Dense
from src.get_weight_dict import get_weight_dict
from src.grid_search.grid_search import grid_search
from src.confusion_matrix import confusion_matrix
from src.class_loss import class_loss
from src.metrics import recall_m, precision_m, f1_m, BalancedSparseCategoricalAccuracy
from tensorflow.keras.metrics import AUC
import torch
import torch.nn as nn

class metabric_model(nn.Module):
    def __init__(self):
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(691, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 28)
        self.fc5 = nn.Linear(28, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)

        return x

class duke_model(nn.Module):
    def __init__(self):
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(74, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class clinical_only:

    def __init__(self, load_model=True):
        self.load_model = load_model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        print("train shape:", X_train.shape)

        if len(y_train.shape) > 1:
            self.multi_target = True
        else:
            self.multi_target = False

        # use shape of data to determine which dataset is being utilized (METABRIC or Duke)
        if X_train.shape[-1] != 691:
            model = duke_model()
            save_path = 'data/saved_models/clinical_duke/keras_clinical_only_model.h5'
        else:
            model = metabric_model()
            save_path = 'data/saved_models/clinical_metabric/keras_clinical_only_model.h5'

        self.criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epochs):
            running_loss = 0.0
            for i in range((X_train.shape[0]-1)//batch_size + 1):
                start_i = i*batch_size
                end_i = start_i+batch_size

                xb = torch.from_numpy(X_train[start_i:end_i]).type(torch.float)
                yb = torch.from_numpy(y_train[start_i:end_i]).type(torch.float)

                pred = model(xb).to(torch.float)
                loss = self.criterion(pred, yb)

                loss.backward()
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

                # print stats
                running_loss += loss.item()
                pred = pred.detach().numpy()
                y_val = torch.from_numpy(y_val).type(torch.float)
                pred = np.argmax(pred, axis=1).astype(np.float)
                accuracy = accuracy_score(y_val, pred)
                f1_score = f1_m(y_val, pred)
                recall = recall_m(y_val, pred)
                balanced_acc = balanced_accuracy_score(y_val, pred)
                print('Completed training batch', epoch, 'Training Loss is: %.4f' %running_loss, 'Accuracy: %.4f' %accuracy, 'F1: %.4f' %f1_score, 'Recall: %.4f' %recall, 'Balanced Accuracy: %.4f' %balanced_acc)
                running_loss = 0.0

        torch.save(self.model.state_dict(), save_path)

        return model

    def test_model(self, X_test, y_test):
        X_test = torch.from_numpy(X_test).type(torch.float)
        y_test = torch.from_numpy(y_test).type(torch.long)
        with torch.no_grad():
            self.model.eval()
            y_pred = self.model(X_test)
            test_loss = self.criterion(y_pred, y_test)

        return results

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=32):
        
        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\keras_clinical_only_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        return self.model
