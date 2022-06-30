from ray.tune.schedulers.async_hyperband import ASHAScheduler
from torch.cuda import is_available
from torch.nn.modules.loss import BCELoss
import numpy as np
from src.grey_to_rgb import grey_to_rgb
from src.confusion_matrix import confusion_matrix
from src.grid_search.grid_search import grid_search
from src.metrics import recall_m, f1_m, BalancedSparseCategoricalAccuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import ray
from ray import tune
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class torch_cnn(nn.Module):
    def __init__(self, num_classes):
        super(torch_cnn, self).__init__()
        self.num_classes = num_classes
        self.to(device)
        self.res = None

    def forward(self, x):
        x = self.res(x)

        return x

    def train_func(self, config, data):
        id_X_train = data[0]
        id_y_train = data[1]
        self.res = data[2]
        X_train = ray.get(id_X_train)
        y_train = ray.get(id_y_train)
        epochs = config['epochs']
        batch_size = config['batch_size']
        lr = config['lr']
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(int(epochs)):
            running_loss = 0.0
            for i in range((X_train.shape[0]-1)//batch_size + 1):
                start_i = i*batch_size
                end_i = start_i+batch_size

                xb = X_train[start_i:end_i]
                yb = y_train[start_i:end_i]

                xb = xb.to(device)
                yb = yb.to(device)

                xb = grey_to_rgb(xb)
                xb = xb/255

                xb = torch.from_numpy(xb)
                xb = torch.reshape(xb, (xb.shape[0], xb.shape[-1], xb.shape[1], xb.shape[2], xb.shape[3]))
                pred = self(xb)

                optimizer = torch.optim.Adam(self.parameters(), lr=lr)

                loss = criterion(pred, yb)
        
                loss.backward()
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

                # print stats
                running_loss += loss.item()
                pred = pred.detach()
                self.loss = running_loss
                pred = np.argmax(pred, axis=1)
                self.accuracy = accuracy_score(yb, pred)
                self.f1_score = f1_m(yb, pred)
                self.recall = recall_m(yb, pred)
                self.balanced_acc = balanced_accuracy_score(yb, pred)
                if i % 2000 == 1999: # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}', 'Accuracy: %.4f' %self.accuracy, 'F1: %.4f' %self.f1_score, 'Recall: %.4f' %self.recall, 'Balanced Accuracy: %.4f' %self.balanced_acc)
                tune.report(loss=running_loss, accuracy=self.accuracy)
                running_loss = 0.0

        print("Finished Training")

        torch.save(self.state_dict(), "torch_cnn_model.pth")

class cnn:

    def __init__(self, load_model=True):
        self.load_model=load_model
        self.res = models.video.r3d_18(pretrained=False)

    def main(self, X_train, y_train, num_samples=10, max_num_epochs=10, gpus_per_trial=2):
        X_train = X_train.type(torch.int8)

        # get number of classes in y
        y = []
        for i in y_train:
            i = int(i)
            y.append(i)

        y = list(set(y))
        num_classes = len(y)

        # make classes start from 0
        class_dict = dict()
        for i in range(num_classes):
            class_dict[y[i]] = i

        i = 0
        for val in y_train:
            new_val = class_dict[int(val)]
            y_train[i] = new_val
            i = i + 1

        if len(y_train.shape) > 1:
            self.multi_target = True
        else:
            self.multi_target = False

        id_X_train = ray.put(X_train)
        id_y_train = ray.put(y_train)

        self.model = torch_cnn(num_classes)
        self.model.to(device)

        config = {
            'epochs':tune.choice([50, 100, 150]),
            'batch_size':tune.choice([8, 16, 32, 64]),
            'lr':tune.loguniform(1e-4, 1e-1),
        }
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2
        )
        if torch.cuda.is_available():
            result = tune.run(
                tune.with_parameters(self.model.train_func, data=[id_X_train, id_y_train, self.res]),
                resources_per_trial={"cpu":14, "gpu":gpus_per_trial},
                config=config,
                metric="loss",
                mode="min",
                num_samples=num_samples,
                scheduler=scheduler
            )
        else:
            result = tune.run(
                tune.with_parameters(self.model.train_func, data=[id_X_train, id_y_train, self.res]),
                resources_per_trial={"cpu":14},
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
            y_pred = self.model(X_test)
            confusion_matrix(y_test, y_pred, save_name="image_only_c_mat_torch")
            test_loss = self.criterion(y_pred, y_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1_score = f1_m(y_test, y_pred)
            recall = recall_m(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)

        return test_loss, accuracy, f1_score, recall, balanced_acc

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=32):

        if self.load_model:
            self.model = torch_cnn()
            self.model.res = self.res
            self.model.load_state_dict(torch.load("torch_cnn_model.pth"), strict=False)
        else:
            self.model = self.main(X_train, y_train)

        return self.model
        