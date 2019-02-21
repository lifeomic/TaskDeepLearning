"""
Extremely simple MNIST example with no test set or anything.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
from torch.utils import data


class Dataset(data.Dataset):

    def __init__(self, features, labels):
        'Initialization'
        self.labels = labels
        self.features = features

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.features[index]
        y = self.labels[index]
        return X, y


class MNIST(nn.Module):

    def __init__(self):
        super(MNIST, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def get_features():
    pd_data = pd.read_csv('mnist_train.csv', delimiter=',', header=None).values
    features = pd_data[:, 1:785].astype(np.float32)
    labels = pd_data[:, 0]
    softmax_labels = np.zeros((len(labels), 10))
    for i, label in enumerate(labels):
        start = np.zeros(10)
        start[label] = 1.0
        softmax_labels[i] = start

    feats = torch.from_numpy(features)
    training_set = Dataset(feats, softmax_labels)
    return data.DataLoader(training_set, batch_size=400)


if __name__ == "__main__":

    device = 'cuda'
    training_generator = get_features()
    mod = MNIST().to(device)

    optimizer = optim.Adam(mod.parameters(), lr=0.001)
    mod.train()
    for i in range(0,20):
        for batch_idx, (data, target) in enumerate(training_generator):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = mod(data)
            loss = F.nll_loss(output, target.argmax(dim=1))
            loss.backward()
            optimizer.step()
        print("Iteration: %s ; loss: %s" % (str(i), str(loss.item())))
    torch.save(mod, 'model_saved')



