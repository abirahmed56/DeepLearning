import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import math

'''
epoch = 1 forward and backward pass of all training samples

batch_size = number of training samples in the forward and backward pass

number of iteration = number of passes, each pass using [batch_size] number of sample

e.g. 100 samples, batch_size --> 100/20 = 5 iteration for 1 epoch
'''


class WineDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])  # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WineDataset()
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

for data in dataloader:
    features, labels = data
    # print(features, data)

# training loop
num_epochs = 2
total_sample = len(dataset)
n_iteration = math.ceil(total_sample / 4)
print(total_sample, n_iteration)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if (i + 1) % 5 == 0:
            print(f"epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_iteration}, inputs {inputs.shape}")

torchvision.datasets.MNIST()
# fasion-mnist, cifar, coco
