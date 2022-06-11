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
        self.ln = nn.LayerNorm(feature_dim)

        self.feature_dim = feature_dim
        self.outputs = dict()

    def forward(self, x):
        h_fc = self.fc(x)
        out = self.ln(h_fc)
        self.outputs['fc'] = h_fc
        self.outputs['ln'] = out
        return out

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        # L.log_param('train_encoder/fc', self.fc, step)
        L.log_sequential('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)

def make_encoder(
    obs_shape, feature_dim, hidden_szs
):
    return Encoder(obs_shape=obs_shape, feature_dim=feature_dim, hidden_szs=hidden_szs)

def main():
    enc = make_encoder(obs_shape=10, feature_dim=5, hidden_szs=[64,64,64])
    print(enc)

if __name__ == "__main__":
    main()