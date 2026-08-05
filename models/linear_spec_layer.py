import torch
import torch.nn as nn

class LinearSpectralLayer(nn.Module):
    """Applique hu @ M pour un opérateur linéaire spectral M (M,M) fixe, non entraînable."""
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer("matrix", matrix)  # (M,M), pas un paramètre entraîné

    def forward(self, hu):
        return hu @ self.matrix