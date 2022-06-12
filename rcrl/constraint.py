import torch.nn as nn
from rcrl.utils import weight_init

def mlp(input_s, output_s, hidden_szs, activation=nn.ReLU):
    sizes = [input_s] + hidden_szs + [output_s]
    layers = []
    for l, r in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(l, r))
        layers.append(activation())

    return nn.Sequential(*layers[:-1])  # remove last unnecessary activation

class ConstraintNetwork(nn.Module):
    def __init__(self, act_dim, constraint_num, hidden_szs=[64,64]):
        super().__init__()    # don't miss the parent class initialization
        self.fc = mlp(act_dim, constraint_num, hidden_szs)
        self.softmax = nn.Softmax(dim=1)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, x):
        x = self.fc(x)
        out = self.softmax(x)
        self.outputs['fc'] = x
        self.outputs['softmax'] = out
        return out

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_constraint/%s_hist' % k, v, step)

        L.log_sequential('train_constraint/fc', self.fc, step)

def main():
    act_dim = 10 
    constraint_num = 3
    net = ConstraintNetwork(act_dim, constraint_num)
    print(net)

if __name__ == "__main__":
    main()