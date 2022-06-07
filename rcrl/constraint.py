import torch
import torch.nn as nn

class Constrain(nn.Module):
    def __init__(self, obs_dim, feature_dim, hidden_szs=[64,64]):
        super().__init__()    # don't miss the parent class initialization
        self.fc = mlp(obs_dim, feature_dim, hidden_szs)

    def forward(self, x):
        return self.fc(x)