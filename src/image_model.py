from ray.tune.schedulers.async_hyperband import ASHAScheduler
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import concatenate
from torch.nn.modules.loss import BCELoss
from torch.autograd import Variable
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

import ray
from ray import tune

device = torch.device('cpu')

class image_clinical(nn.Module):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        super().__init__()
        self.to(device)
        self.relu = nn.ReLU()
        self.clinical_track()
        self.image_track()
        self.to(device)

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

    def train_func(self, config):
        print(config)
        X_train = self.X_train
        y_train = self.y_train
        epochs = config['epochs']
        batch_size = config['batch_size']
        lr = config['lr']
        for epoch in range(int(epochs)):
            running_loss = 0.0
            for i in range((X_train[0][1].shape[0]-1)//batch_size + 1):

                start_i = i*batch_size
                end_i = start_i+batch_size

                xb = [torch.from_numpy(X_train[0][0][start_i:end_i]).to(device), torch.from_numpy(X_train[0][1][start_i:end_i]).to(device)]
                yb = torch.from_numpy(y_train[start_i:end_i]).type(torch.float).to(device)
                yb = Variable(yb)

                xb[1] = torch.reshape(xb[1], (-1, 1, 256, 256))
                xb[1] = xb[1].type(torch.float)
                xb[0] = xb[0].type(torch.float)
                yb = yb.type(torch.float)
                pred = self(xb)

                criterion = torch.nn.BCEWithLogitsLoss()
                optimizer = torch.optim.Adam(self.parameters(), lr=lr)

                yb = yb.flatten()
                pred = pred.flatten()
                loss = criterion(pred, yb)
        
                loss.backward()
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

                # print stats
                running_loss += loss.item()
                pred = pred.detach().numpy()
                yb = np.asarray(yb).astype(np.float)
                self.loss = running_loss
                pred = pred.flatten()
                pred = pred.round().astype(np.float)
                self.accuracy = accuracy_score(yb, pred)
                self.f1_score = f1_m(yb, pred)
                self.recall = recall_m(yb, pred)
                self.balanced_acc = balanced_accuracy_score(yb, pred)
                if i % 2000 == 1999: # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}', 'Accuracy: %.4f' %self.accuracy, 'F1: %.4f' %self.f1_score, 'Recall: %.4f' %self.recall, 'Balanced Accuracy: %.4f' %self.balanced_acc)
                tune.report(loss=running_loss, accuracy=self.accuracy)
                running_loss = 0.0

        print("Finished Training")

        #torch.save(self.state_dict(), "..\\data\\saved_models\\image_clinical\\torch_image_clinical_model.h5")

class image_model:

    def __init__(self, load_model=True):
        self.load_model = load_model

    def main(self, X_train, y_train, num_samples=10, max_num_epochs=10, gpus_per_trial=2):

        if len(y_train.shape) > 1:
            self.multi_target = True
        else:
            self.multi_target = False

        self.model = image_clinical(X_train, y_train)
        self.model.to(device)

        config = {
            'epochs':tune.choice([50, 100, 150]),
            'batch_size':tune.choice([8, 16, 32, 64]),
            'lr':tune.loguniform(1e-4, 1e-1)
        }      
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)
        result = tune.run(
            tune.with_parameters(self.model.train_func),
            resources_per_trial={"cpu":4},
            config=config,
            metric="loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler
        )

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result['accuracy']))

        self.model.train_func(config=best_trial.config)

        return self.model

    def test_model(self, X_test, y_test, trial):
        X_test[0][1] = np.reshape(X_test[0][1], (X_test[0][1].shape[0], 1, 256, 256))
        X_test = [torch.from_numpy(X_test[0][0]).type(torch.float), torch.from_numpy(X_test[0][1]).type(torch.float)]
        y_test = torch.from_numpy(y_test).type(torch.float)
        with torch.no_grad():
            self.model.eval()
            y_pred = self.model(X_test)
            y_pred = y_pred.round()
            confusion_matrix(y_test, y_pred, save_name="image_only_c_mat_torch")
            test_loss = self.criterion(y_pred, y_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1_score = f1_m(y_test, y_pred)
            recall = recall_m(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)

        return test_loss, accuracy, f1_score, recall, balanced_acc

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=128):
        
        if self.load_model:
            self.model = torch.load('data\\saved_models\\torch_image_clinical_model.h5')
        else:
            self.model = self.main(X_train, y_train)

        return self.model
        