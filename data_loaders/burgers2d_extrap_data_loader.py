import numpy as np
import torch
from basis.fourier2d import FourierBasis2D
from utilities.grid import Grid3DToTensors, Grid3DPINNsFormerToTensors

class Burgers2DExtGroundTruthLoader:
    def __init__(self, model_name="NeuSA"):
        self.model_name = model_name

        self.x, self.y, self.t, self.u, self.v = generate_ground_truth()
        
    def get_ground_truth_grid(self):
        return self.x,self.y,self.t

    def get_ground_truth_solution_u(self):

        return self.u
    
    def get_ground_truth_solution_v(self):

        return self.v
    
    def get_ground_truth_grid_as_tensors(self):

        if self.model_name == "PINNsFormer":
            grid_as_burgers2d_groundtruth_generatortensors = Grid3DPINNsFormerToTensors(self.x,self.y, self.t)  
            x_tensor = grid_as_tensors.aug_grid_as_column_x
            y_tensor = grid_as_tensors.aug_grid_as_column_y
            t_tensor = grid_as_tensors.aug_grid_as_column_t

        else:
            
            grid_as_tensors = Grid3DToTensors(self.x,self.y,self.t)

            x_tensor = grid_as_tensors.x
            y_tensor = grid_as_tensors.y
            t_tensor = grid_as_tensors.t

        return x_tensor, y_tensor, t_tensor
    
    def get_grid_size(self):
        return len(self.x), len(self.y), len(self.t)

def generate_ground_truth():
    
    domain_x = [0.0,4.0]
    domain_y = [0.0,4.0]
    domain_t = [0.0,2.0]
    time_steps = 401
    Nx = 201
    Ny = 201
    u, v = burgers2d_groundtruth_generator(domain_t, domain_x, domain_y, time_steps)
    t = np.linspace(domain_t[0], domain_t[1], time_steps)
    x = np.linspace(domain_x[0], domain_x[1], Nx+1)[:-1]
    y = np.linspace(domain_y[0], domain_y[1], Ny+1)[:-1]

    return x, y, t, u.flatten(), v.flatten()


def rk4(u, v, dt, f, g):
        
        k1 = dt * f(u, v)
        l1 = dt * g(u, v)

        k2 = dt * f(u + 0.5 * k1, v + 0.5 * l1)
        l2 = dt * g(u + 0.5 * k1, v + 0.5 * l1)

        k3 = dt * f(u + 0.5 * k2, v + 0.5 * l2)
        l3 = dt * g(u + 0.5 * k2, v + 0.5 * l2)

        k4 = dt * f(u + k3, v + l3)
        l4 = dt * g(u + k3, v + l3)

        u_next = u + (k1 + 2*k2 + 2*k3 + k4) / 6
        v_next = v + (l1 + 2*l2 + 2*l3 + l4) / 6

        return u_next, v_next

def burgers2d_groundtruth_generator(domain_t=(0.,1.), domain_x=(0.,4.), domain_y=(0.,4.), time_steps=201):

    M = 201
    N = 201

    ts = torch.linspace(*domain_t,time_steps)

    # basis and grid
    basis = FourierBasis2D(M,N,domain_x,domain_y)
    X,Y = torch.meshgrid(basis.x,basis.y,indexing='ij')
    grid = torch.stack((X,Y),dim=-1)    # shape = (M,N,2)
    xs = grid[:,:,0:1]
    ys = grid[:,:,1:2]

    # initial condition
    u0 = (torch.sin(np.pi*xs)*torch.sin(np.pi*ys)).squeeze(-1)    # shape = (M,N)
    v0 = (torch.cos(np.pi*xs)*torch.cos(np.pi*ys)).squeeze(-1)    # shape = (M,N)
    
    def fu(u, v):
        
        hu = basis.dbt(u)
        u_x = basis.idbt(torch.einsum('ik,...kj->...ij',basis.Dx,hu))
        u_y = basis.idbt(torch.einsum('...ik,kj->...ij',hu,basis.Dy))
        nuD2u = basis.idbt(0.01 * basis.D2 * hu)

        return nuD2u - u * u_x - v * u_y

    def fv(u, v):
        
        hv = basis.dbt(v)
        v_x = basis.idbt(torch.einsum('ik,...kj->...ij',basis.Dx,hv))
        v_y = basis.idbt(torch.einsum('...ik,kj->...ij',hv,basis.Dy))
        nuD2v = basis.idbt(0.01 * basis.D2 * hv)

        return nuD2v - u * v_x - v * v_y

    T = [0.0]
    U = [u0]
    V = [v0]
    u = u0
    v = v0
    dt = ts[1] - ts[0]
    for t in ts[1:]:
        u,v = rk4(u,v,dt,fu,fv)

        T.append(t)
        U.append(u)
        V.append(v)

    
    U_tensor = torch.stack(U)
    u_gt = U_tensor.detach().cpu().numpy()
    V_tensor = torch.stack(V)
    v_gt = V_tensor.detach().cpu().numpy()
    
    return u_gt, v_gt
    