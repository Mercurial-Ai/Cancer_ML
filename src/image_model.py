import struct
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
from src.grey_to_rgb import grey_to_rgb
from tensorflow.keras.metrics import AUC
import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models
from src.resnet import resnet18
import numpy as np
import ray
from ray import tune

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class image_clinical(nn.Module):
    def __init__(self, num_classes, res):
        super(image_clinical, self).__init__()
        self.res = res
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.clinical_track()

        self.fc1_cat = nn.Linear(415, 26)
        self.fc2_cat = nn.Linear(26, num_classes)

    def clinical_track(self):
        self.fc1 = nn.Linear(603, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 15)

    def forward(self, image_data, clinical_data):

        # clinical
        clinical_x = self.relu(self.fc1(clinical_data))
        clinical_x = self.relu(self.fc2(clinical_x))
        clinical_x = self.relu(self.fc3(clinical_x))

        # image
        image_x = self.res(image_data)

        x = torch.cat([clinical_x, image_x], dim=1)
        
        x = self.relu(self.fc1_cat(x))
        x = self.fc2_cat(x)

        return x

def train_func(config, data):
    model = data[0]
    id_X_train_image = data[0]
    id_X_train_clinical = data[1]
    id_y_train = data[2]
    X_train_image = ray.get(id_X_train_image)
    X_train_clinical = ray.get(id_X_train_clinical)
    y_train = ray.get(id_y_train)
    epochs = config['epochs']
    batch_size = config['batch_size']
    lr = config['lr']
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(int(epochs)):
        running_loss = 0.0
        for i in range((X_train_clinical.shape[0]-1)//batch_size + 1):

            start_i = i*batch_size
            end_i = start_i+batch_size

            if type(y_train) == np.ndarray:
                y_train = torch.from_numpy(y_train)

            image_data = X_train_image[start_i:end_i]
            clinical_data = X_train_clinical[start_i:end_i].to(device)

            yb = y_train[start_i:end_i].to(device)
            yb = Variable(yb)

            image_data = torch.unsqueeze(image_data.type(torch.float), -1)
            image_data = grey_to_rgb(image_data).to(device)/255
            # reshape to have 3 channels
            image_data = torch.reshape(image_data, (image_data.shape[0], image_data.shape[-1], image_data.shape[1], image_data.shape[2], image_data.shape[3])).type(torch.float)
            clinical_data = clinical_data.type(torch.float)

            # concatenate image and clinical data together to input into forward function
            pred = model(image_data, clinical_data)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            loss = criterion(pred, yb)
    
            loss.backward()
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            # print stats
            running_loss += loss.item()
            pred = pred.detach()
            loss = running_loss
            pred = torch.argmax(pred, axis=1)
            pred = pred.cpu()
            yb = yb.cpu()
            accuracy = accuracy_score(yb, pred)
            f1_score = f1_m(yb, pred)
            recall = recall_m(yb, pred)
            balanced_acc = balanced_accuracy_score(yb, pred)
            if i % 2000 == 1999: # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}', 'Accuracy: %.4f' %accuracy, 'F1: %.4f' %f1_score, 'Recall: %.4f' %recall, 'Balanced Accuracy: %.4f' %balanced_acc)
            tune.report(loss=running_loss, accuracy=accuracy)
            running_loss = 0.0

        torch.cuda.empty_cache()

    print("Finished Training")

class image_model:

    def __init__(self, load_model=True):
        self.load_model = load_model
        self.res = models.video.r3d_18(pretrained=False)

    def main(self, X_train, y_train, num_samples=10, max_num_epochs=10, gpus_per_trial=2):
        # get number of classes in y
        y = []
        for i in y_train:
            i = int(i)
            y.append(i)
        
        y = list(set(y))
        self.num_classes = len(y)

        # make classes start from 0
        class_dict = dict()
        for i in range(self.num_classes):
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

        self.model = image_clinical(self.num_classes, self.res)
        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), "gpus!")
            self.model = nn.DataParallel(self.model)

        self.model.to(device)

        config = {
            'epochs':tune.choice([50, 100, 150]),
            'batch_size':tune.choice([1, 2, 3, 4]),
            'lr':tune.loguniform(1e-4, 1e-1)
        }      
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=3)
        if torch.cuda.is_available():
            result = tune.run(
                tune.with_parameters(train_func, data=[id_X_train, id_y_train, self.res]),
                resources_per_trial={"cpu":14, "gpu":gpus_per_trial},
                config=config,
                metric="accuracy",
                mode="min",
                num_samples=num_samples,
                scheduler=scheduler
            )
        else:
            result = tune.run(
                tune.with_parameters(train_func, data=[id_X_train, id_y_train, self.res]),
                resources_per_trial={"cpu":14},
                config=config,
                metric="accuracy",
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

        self.model.train_func(config=best_trial.config, data=[id_X_train, id_y_train, self.res])

        torch.save(self.model.state_dict(), "torch_image_clinical_model.pth")

        return self.model

    def test_model(self, X_test, y_test):
        X_test = [torch.from_numpy(np.array(X_test[0])).type(torch.float), torch.from_numpy(np.array(X_test[1])).type(torch.float)]
        X_test[0] = torch.unsqueeze(X_test[0], -1)
        X_test[0] = grey_to_rgb(X_test[0])/255
        # reshape to have 3 channels
        X_test[0] = np.reshape(X_test[0], (X_test[0].shape[0], X_test[0].shape[-1], X_test[0].shape[1], X_test[0].shape[2], X_test[0].shape[3]))
        i = 0
        for arr in X_test:
            if type(arr) == np.ndarray:
                arr = torch.from_numpy(arr).type(torch.float)
                X_test[i] = arr
            i = i + 1
        self.criterion = torch.nn.CrossEntropyLoss()
        y_test = torch.from_numpy(np.array(y_test)).type(torch.float)
        with torch.no_grad():
            y_pred = self.model(X_test)
            confusion_matrix(y_test, y_pred, save_name="image_only_c_mat_torch")
            test_loss = self.criterion(y_pred, y_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1_score = f1_m(y_test, y_pred)
            recall = recall_m(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)

        return test_loss, accuracy, f1_score, recall, balanced_acc

    def get_model(self, X_train=None, y_train=None, X_val=None, y_val=None, epochs=10, batch_size=128):
        
        if self.load_model:
            self.model = image_clinical(self.num_classes, self.res)
            self.model.load_state_dict(torch.load("torch_image_clinical_model.pth"), strict=False)
        else:
            self.model = self.main(X_train, y_train)

        return self.model
        