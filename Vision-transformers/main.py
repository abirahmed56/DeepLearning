import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tqdm import tqdm


# Hyper parameter
RANDOM_SEED = 42
BATCH_SIZE = 512
EPOCH = 40
LEARNING_RATE = 1e-4
NUM_CLASSES = 10
PATCH_SIZE = 4
IMG_SIZE = 28
IN_CHANNEL = 1
NUM_HEADS = 8
DROPOUT = .001
HIDDEN_DIM = 768
ADAM_EWIGHT_DECAY = 0
ADAM_BETAS = (.9, .999)
ACTIVATION = 'gelu'
NUM_ENCODERS = 4
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNEL # 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE ) ** 2 # 49

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

class PatchEmbeddings(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super.__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2)
        )
        self.cls_token = nn.Parameter(torch.randn(size=(1, in_channels, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings + x
        x = self.dropout(x)
        return x
    
model = PatchEmbeddings(EMBED_DIM, PATCH_SIZE, NUM_PATCHES, DROPOUT, IN_CHANNEL).to(device)
x = torch.randn(512, 1, 28, 28).to(device)
print(model(x).shape)
