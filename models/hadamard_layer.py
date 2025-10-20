import torch
import torch.nn as nn

class HadamardLayer(nn.Module):
    def __init__(self,multipliers):
        super().__init__()
        self.register_buffer("multipliers", torch.tensor(multipliers))

    def forward(self,x):
        return x * self.multipliers