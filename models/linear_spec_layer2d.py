import torch
import torch.nn as nn

class LaplacianSpectralLayer2D(nn.Module):
    def __init__(self, D2x, D2y):
        super().__init__()
        self.register_buffer("D2x", D2x)
        self.register_buffer("D2y", D2y)

    def forward(self, hu):
        return 0.01 * (self.D2x @ hu + hu @ self.D2y.T)