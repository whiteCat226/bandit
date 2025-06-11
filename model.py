## model.py
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 9)  # 9 classes in PathMNIST
        )

    def forward(self, x):
        return self.net(x)
