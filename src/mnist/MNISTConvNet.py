import torch
import torch.nn as nn

from utils.init_utils import initialize_weights

class MNISTConvNet(nn.Module):
    def __init__(self):
        self.layer_1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=2) # C, H, W = 3, 4, 4
        self.classifier = nn.Linear(48, 10)

    def forward(self, x):
        print(x.size())
        x = self.layer_1(x)
        x = nn.functional.relu(x)
        x = x.flatten(start_dim=1)
        y_hat = self.classifier(x)
        return y_hat
