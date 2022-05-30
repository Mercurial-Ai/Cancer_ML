from ray.tune.schedulers.async_hyperband import ASHAScheduler
from torch.nn.modules.loss import BCELoss
import numpy as np
from src.confusion_matrix import confusion_matrix
from src.grid_search.grid_search import grid_search
from src.metrics import recall_m, f1_m, BalancedSparseCategoricalAccuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from ray import tune

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class torch_cnn(nn.Module):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        super(torch_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

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

    def train_func(self, config):
        X_train = self.X_train
        y_train = self.y_train
        epochs = config['epochs']
        batch_size = config['batch_size']
        lr = config['lr']
        for epoch in range(int(epochs)):
            running_loss = 0.0
            for i in range((X_train.shape[0]-1)//batch_size + 1):
                start_i = i*batch_size
                end_i = start_i+batch_size

                xb = X_train[start_i:end_i]
                yb = y_train[start_i:end_i]

                xb_shape = xb.shape

                xb = torch.from_numpy(xb)
                yb = torch.from_numpy(yb).type(torch.float)

                xb = xb.to(device)
                yb = yb.to(device)

                xb = torch.reshape(xb, (xb_shape[0], 1, 256, 256))
                xb = xb.type(torch.float)
                pred = self(xb)

                criterion = torch.nn.BCEWithLogitsLoss()
                optimizer = torch.optim.Adam(self.parameters(), lr=lr)

                pred = pred.flatten()
                yb = yb.flatten()
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
                pred = pred.astype(np.float)
                try:
                    self.accuracy = accuracy_score(yb, pred)
                    self.f1_score = f1_m(yb, pred)
                    self.recall = recall_m(yb, pred)
                    self.balanced_acc = balanced_accuracy_score(yb, pred)
                except:
                    pred = pred.round()
                    self.accuracy = accuracy_score(yb, pred)
                    self.f1_score = f1_m(yb, pred)
                    self.recall = recall_m(yb, pred)
                    self.balanced_acc = balanced_accuracy_score(yb, pred)
                if i % 2000 == 1999: # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}', 'Accuracy: %.4f' %self.accuracy, 'F1: %.4f' %self.f1_score, 'Recall: %.4f' %self.recall, 'Balanced Accuracy: %.4f' %self.balanced_acc)
                tune.report(loss=running_loss, accuracy=self.accuracy)
                running_loss = 0.0

        print("Finished Training")

        #torch.save(self.state_dict(), "..\\data\\saved_models\\image_only\\torch_cnn_model.h5")

class cnn:

    def __init__(self, load_model=True):
        self.load_model=load_model

    def main(self, X_train, y_train, num_samples=10, max_num_epochs=10, gpus_per_trial=2):

        if len(y_train.shape) > 1:
            self.multi_target = True
        else:
            self.multi_target = False

        self.model = torch_cnn(X_train, y_train)
        self.model.to(device)

        config = {
            'epochs':tune.choice([50, 100, 150]),
            'batch_size':tune.choice([8, 16, 32, 64]),
            'lr':tune.loguniform(1e-4, 1e-1)
        }
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2
        )
        result = tune.run(
            tune.with_parameters(self.model.train_func),
            resources_per_trial={"cpu":4, "gpu":gpus_per_trial},
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

    def test_model(self, X_test, y_test):
        self.criterion = torch.nn.BCEWithLogitsLoss()
        X_test = np.reshape(X_test, (X_test.shape[0], 1, 256, 256))
        X_test = torch.from_numpy(X_test).type(torch.float)
        y_test = torch.from_numpy(y_test).type(torch.float)
        with torch.no_grad():
            self.model.eval()
            y_pred = self.model(X_test)
            confusion_matrix(y_test, y_pred, save_name="image_only_c_mat_torch")
            y_pred = y_pred.flatten()
            y_pred = torch.abs(torch.round(y_pred))
            test_loss = self.criterion(y_pred, y_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1_score = f1_m(y_test, y_pred)
            recall = recall_m(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)

        return test_loss, accuracy, f1_score, recall, balanced_acc

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=32):

        if self.load_model:
            self.model = torch.load('data\\saved_models\\torch_cnn_model.h5')
        else:
            self.model = self.main(X_train, y_train)

        return self.model
        