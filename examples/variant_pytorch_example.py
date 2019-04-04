import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torch.utils import data
import json
from sklearn.model_selection import train_test_split


class Dataset(data.Dataset):

    def __init__(self, features, labels):
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.features[index]
        y = self.labels[index]
        return X, y


class VariantModel(nn.Module):

    def __init__(self):
        super(VariantModel, self).__init__()
        self.fc1 = nn.Linear(732, 512)
        self.do1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.do2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.do1(x)
        x = F.relu(self.fc2(x))
        x = self.do2(x)
        x = self.fc3(x)
        return F.sigmoid(x)


def train_test_data_loader(test_split=0.1):
    with open('variant_data.json', 'r') as f:
        loaded_data = json.load(f)
    labels = np.asarray(loaded_data['labels'])
    features = np.asarray(loaded_data['features']).astype(np.float32)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                                test_size=test_split,
                                                                                random_state=12345)
    training_set = Dataset(features_train, labels_train)
    test_set = Dataset(features_test, labels_test)
    return data.DataLoader(training_set, batch_size=256), data.DataLoader(test_set)


def train_model(model, training_generator, epochs=25):
    optimizer = optim.Adam(mod.parameters(), lr=0.009)
    model.train()

    for i in range(0, epochs):
        for (data, target) in training_generator:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, target.type(torch.FloatTensor))
            loss.backward()
            optimizer.step()
        print("Iteration: %s ; loss: %s" % (str(i), str(loss.item())))


if __name__ == "__main__":
    device = 'cpu'
    training_generator, test_generator = train_test_data_loader(0.15)
    mod = VariantModel().to(device)
    train_model(mod, training_generator, epochs=25)

