# baseline implementation of PINNs
# paper: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
# link: https://www.sciencedirect.com/science/article/pii/S0021999118307125
# code: https://github.com/maziarraissi/PINNs

import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, num_layer=4):
        super().__init__()

        self.out_dim = out_dim

        layers = []
        for i in range(num_layer):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, *args: torch.Tensor):
        src = torch.cat(args, dim=-1)
        return self.linear(src)
