# baseline implementation of First Layer Sine
# paper: Learning in Sinusoidal Spaces with Physics-Informed Neural Networks
# link: https://arxiv.org/abs/2109.09338

import torch
import torch.nn as nn



class SinAct(nn.Module):
    def __init__(self):
        super(SinAct, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class FLS(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, num_layer=4):
        super().__init__()

        self.out_dim=out_dim

        layers = []
        for i in range(num_layer):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(SinAct())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)


    def forward(self, *args: torch.Tensor):
        src = torch.cat(args, dim=-1)
        return self.linear(src)