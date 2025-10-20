import torch
import torch.nn as nn

class HeatLikeVectorField(nn.Module):
    def __init__(self, F_neural, F0, model_weight):
        super().__init__()
        self.F_neural = F_neural
        self.F0 = F0
        self.model_weight = model_weight
    
    def forward(self,x):
        return self.model_weight * self.F_neural(x) + self.F0(x)
    
class HeatLikeVectorFieldWith2DOutput(nn.Module):
    # This implements the vector field of an ODE corresponding to a PDE having the form
                        # u_t = D2u + non-linear terms
                        # v_t = D2v + non-linear terms
    # F0 stands for the initial vector field, which just computes the Laplacian
    # the neural networks for u and v learn the corresponding non-linearities.

    def __init__(self,F_neural_u,F_neural_v,F0,model_weight):
        super().__init__()
        self.F_neural_u = F_neural_u
        self.F_neural_v = F_neural_v
        self.F0 = F0
        self.model_weight = model_weight

    def forward(self,x):
        hu,hv = x.chunk(2,dim=-1) # u and v have shape (M,N)
        Fhu = self.model_weight * self.F_neural_u(x) + self.F0(hu)
        Fhv = self.model_weight * self.F_neural_v(x) + self.F0(hv)

        return torch.cat([Fhu,Fhv], dim=-1)


class WaveLikeVectorField(nn.Module):
    def __init__(self, F_neural, F0, model_weight):
        super().__init__()
        self.F_neural = F_neural
        self.F0 = F0
        self.model_weight = model_weight

    def forward(self,huv):
        hu,hv = huv.chunk(2,dim=-1)
        hu_t = hv
        hv_t = self.model_weight * self.F_neural(hu) + self.F0(hu)
        return torch.cat([hu_t,hv_t], dim=-1)