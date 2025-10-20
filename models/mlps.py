import torch
import torch.nn as nn

class TransposeLayer2D(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        # (a,b) ---> (b,a)
        x = x.transpose(-2,-1)
        return x

class TransposeLayer3D(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        # (a,b,c) ---> (c,a,b)
        x = x.transpose(-2,-1).transpose(-3,-2)
        return x
 
class MLP:
    def __new__(
        cls,
        in_dim,
        hidden_dim,
        out_dim,
        num_hidden_layers,
        activation=nn.LeakyReLU(),
        bias=True
    ) -> nn.Sequential:

        layers = []

        layers += [nn.Linear(in_dim,hidden_dim,bias=bias), activation]

        for _ in range(num_hidden_layers-1):
            layers += [nn.Linear(hidden_dim,hidden_dim,bias=bias), activation]

        layers += [nn.Linear(hidden_dim,out_dim,bias=bias)]

        return nn.Sequential(*layers)

class MLPWithTranspose2D:
    # usually, the input of an MLP is a matrix (M,N), where each line
    # corresponds to, e.g., a grid point. Thus, the input lines are independent, in some sense.
    # However, when the input is a 2D grid (M,N), then the lines are not independent.
    # A usual MLP would not be fully connected, in this case. To see this, suppose that the first layer
    # is (N,P). Then, this layer performs a matrix multiplication (M,N)x(N,P), where the first line of the input
    # only "appears" in the first line of the output.
    # to overcome this problem, we insert layers which transpose the outputs of the respective previous layers.
    
    def __new__(
        cls,
        M,
        N,
        widen_factor=2,
        activation=nn.LeakyReLU(),
        bias=False,
        num_additional_hidden_layers=0,
        additional_hidden_layer_bias=True,
    ) -> nn.Sequential:

        # the input is an (M,N) matrix
        
        layers = []
                                                                            # c = widen_factor
        layers += [nn.Linear(N,widen_factor*N,bias=bias),TransposeLayer2D()] # (M,N) x (N,cN) ---> (M,cN) ----> (cN,M)
        layers += [nn.Linear(M,widen_factor*M,bias=bias),TransposeLayer2D(),activation] # (cN,M) x (M,cM) ---> (cN,cM) ---> (cM,cN)

        for _ in range(num_additional_hidden_layers):
            layers += [nn.Linear(widen_factor*N,widen_factor*N,bias=additional_hidden_layer_bias),activation] # (cM,cN) x (cN,cN) ---> (cM,cN)
        
        layers += [nn.Linear(widen_factor * N,N,bias=bias),TransposeLayer2D()] # (cM,cN) x (cN,N) ---> (cM,N) ---> (N,cM)
        layers += [nn.Linear(widen_factor* M,M,bias=bias),TransposeLayer2D()] #  (N,cM) x (cM,M) ---> (N,M) ---> (M,N)

        return nn.Sequential(*layers)

class MLPWithTranspose3D:
    # This is analogous to MLPWithTranspose2D, but for 3D spatial grids.
    def __new__(
            cls,
            M,
            N,
            P,
            widen_factor=1,
            activation=nn.LeakyReLU(),
            bias=False,
            num_additional_hidden_layers=0,
            additional_hidden_layer_bias=False,
    ) -> nn.Sequential:
        
        layers = []
                                                                                 # c = widen_factor
        layers += [nn.Linear(P,widen_factor * P,bias=bias),TransposeLayer3D()]  # (M,N,P) x (P, cP) ---> (M,N,cP) ---> (cP,M,N)
        layers += [nn.Linear(N,widen_factor * N,bias=bias),TransposeLayer3D()] # (cP,M,N) x (N,cN) ---> (cP,M,cN) ---> (cN,cP,M)
        layers += [nn.Linear(M,widen_factor * M,bias=bias),TransposeLayer3D(),activation] # (cN,cP,M) x (M,cM) ---> (cN,cP,cM) ---> (cM,cN,cP)

        for _ in range(num_additional_hidden_layers):
            layers += [nn.Linear(widen_factor * P,widen_factor * P,bias=additional_hidden_layer_bias),activation]
        
        layers += [nn.Linear(widen_factor * P,P,bias=bias),TransposeLayer3D()]  # (cM,cN,cP) x (cP,P) ---> (cM,cN,P) ---> (P,cM,cN)
        layers += [nn.Linear(widen_factor * N,N,bias=bias),TransposeLayer3D()]  # (P,cM,cN) x (cN,N) ---> (P,cM,N) ---> (N,P,cM)
        layers += [nn.Linear(widen_factor * M,M,bias=bias),TransposeLayer3D()]  # (N,P,cM) x (cM,M) ---> (N,P,M) ---> (M,N,P)

        return nn.Sequential(*layers)


class MLPWithTranspose2DWith2DOutput:
    def __new__(
            cls,
            M,
            N,
            widen_factor=2,
            activation=nn.LeakyReLU(),
            bias=False,
            num_additional_hidden_layers=0,
            additional_hidden_layer_bias=False,
    ) -> nn.Sequential:
        
        # The input is a tensor (M,2N) and the output is a tensor (M,N).
        # This is applied for the Burgers 2D equation, whose output is (u,v).
        # The corresponding ODE has the form (u',v') = (F1(u,v),F2(u,v)), so we use two neural networks with the same architecture.
        # In the case of the Wave equation, the ODE has the form (u',v') = (v,F(u)), so the input of the neural network was (M,N)

        layers = []                                                                 # c = widen_factor
        layers += [nn.Linear(2*N,widen_factor * N,bias=bias),TransposeLayer2D()]    # (M,2N) x (2N,cN) --> (M,cN) ---> (cN,M)
        layers += [nn.Linear(M,widen_factor * M,bias=bias),TransposeLayer2D(),activation]   # (cN,M) x (M,cM) ---> (cN,cM) ---> (cM,cN)
        layers += [nn.Linear(widen_factor * N,widen_factor * N,bias=bias),TransposeLayer2D()]  # (cM,cN) x (cN,cN) ---> (cM,cN) ---> (cN,cM)
        layers += [nn.Linear(widen_factor*M,widen_factor * M,bias=bias),TransposeLayer2D(),activation]    # (cN,cM) x (cM,cM) ---> (cN,cM) ---> (cM,cN)

        for _ in range(num_additional_hidden_layers):
            layers += [nn.Linear(widen_factor * N, widen_factor * N,bias=additional_hidden_layer_bias),TransposeLayer2D(),activation]

        layers += [nn.Linear(widen_factor * N,N,bias=bias),TransposeLayer2D()]  # (cM,cN) x (cN,N) ---> (cM,N) ---> (N,cM)
        layers += [nn.Linear(widen_factor * M,M,bias=bias),TransposeLayer2D()]  # (N,cM) x (cM,M) ---> (N,M) ---> (M,N)

        return nn.Sequential(*layers)
