import torch
import torch.nn as nn

def mlp(input_s, output_s, hidden_szs, activation=nn.ReLU):
    sizes = [input_s] + hidden_szs + [output_s]
    layers = []
    for l, r in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(l, r))
        layers.append(activation())

    return nn.Sequential(*layers[:-1])  # remove last unnecessary activation

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, hidden_szs=[64,64]):
        super().__init__()    # don't miss the parent class initialization
        self.fc = mlp(obs_shape, feature_dim, hidden_szs)
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.fc(x)

def make_encoder(
    obs_shape, feature_dim, hidden_szs
):
    return Encoder(obs_shape=obs_shape, feature_dim=feature_dim, hidden_szs=hidden_szs)

def main():
    enc = make_encoder(obs_shape=10, feature_dim=5, hidden_szs=[64,64,64])
    print(enc)

if __name__ == "__main__":
    main()