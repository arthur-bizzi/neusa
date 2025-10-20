from data_loaders.burgers2d_extrap_data_loader import Burgers2DExtGroundTruthLoader
import argparse
import torch
from torchdyn.core import NeuralODE
import numpy as np
from utilities.eval_chunks import eval_chunked
import os
from models.PINN import PINN
from models.QRes import QRes
from models.FLS import FLS
from basis.fourier2d import FourierBasis2D
from models.mlps import MLPWithTranspose2DWith2DOutput
from models.vector_fields import HeatLikeVectorFieldWith2DOutput
from models.hadamard_layer import HadamardLayer
from utilities.plot import plot_comparison

def time_rRMSE(u_gt, u_pred):
    # u_gt and u_pred are both ntxnxxny tensors
    # returns a nt tensor containing the rRMSE for each time instant
    error = u_pred-u_gt
    square_error = np.sum(error**2, axis=(1,2))
    base = np.sum(u_gt**2, axis=(1,2))
    return np.sqrt(square_error/base)

if __name__ == "__main__":
    # Extrapolates the models for 2D Burgers trained on the [0,1] interval. 
    # Extrapolates and evaluates on the [0,2] time interval
    parser = argparse.ArgumentParser("extrapolation_experiment")
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder is saved
    parser.add_argument('--seed', type=int,default=42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    dir_name = args.dir
    seed = args.seed
    device = args.device


    data_loader = Burgers2DExtGroundTruthLoader()

    u_gt = data_loader.get_ground_truth_solution_u()
    v_gt = data_loader.get_ground_truth_solution_v()
    x, y, t = data_loader.get_ground_truth_grid_as_tensors()
    x = x.to(device)
    y = y.to(device)
    t = t.to(device)
    nx, ny, nt = data_loader.get_grid_size()
    u_gt = u_gt.reshape((nt,nx,ny))
    v_gt = v_gt.reshape((nt,nx,ny))

    extrapolated_u = {}
    extrapolated_v = {}
    extrapolated_error = {}

    # Evaluating the extrapolation of NeuSA
    model_name = "NeuSA"
    # domain and resolution
    domain_x = [0.0,4.0]
    domain_y = [0.0,4.0]
    domain_t = [0.0,1.0]
    M = 201
    N = 201
    time_steps = 201
    ts = torch.linspace(*domain_t,time_steps)
    basis = FourierBasis2D(M, N, domain_x=domain_x, domain_y=domain_y, device=device)
    # initial condition
    X,Y = torch.meshgrid(basis.x,basis.y,indexing='ij')
    grid = torch.stack((X,Y),dim=-1)    # shape = (M,N,2)
    xs = grid[:,:,0:1]
    ys = grid[:,:,1:2]
    _, _, ts = data_loader.get_ground_truth_grid()
    ts = torch.tensor(ts, device=device)
    u0 = (torch.sin(np.pi*xs)*torch.sin(np.pi*ys)).squeeze(-1).to(device)    # shape = (M,N)
    v0 = (torch.cos(np.pi*xs)*torch.cos(np.pi*ys)).squeeze(-1).to(device)    # shape = (M,N)
    hu0 = basis.dbt(u0)
    hv0 = basis.dbt(v0)
    huv0 = torch.cat([hu0,hv0],dim=-1).to(device)  # shape = (M,2N)
    
    # Defining the different components of the NeuSA vector field
    omegas = 0.01*basis.D2
    F0 = HadamardLayer(omegas)
    F_neural_u = MLPWithTranspose2DWith2DOutput(M=M,N=N)
    F_neural_v = MLPWithTranspose2DWith2DOutput(M=M,N=N)
    F = HeatLikeVectorFieldWith2DOutput(F_neural_u,F_neural_v,F0,model_weight=0.1).to(device)
    NODE = NeuralODE(F,solver='rk4',sensitivity='autograd',return_t_eval=False)
    
    # Reading the saved file
    model_path = os.path.join("results", "burgers2d", dir_name, "NeuSA", f"seed_{seed}", "model.pt")
    F.load_state_dict(torch.load(model_path, weights_only=True))

    # Evaluating
    with torch.no_grad():
        huv = NODE(huv0, ts)
    hu,hv = huv.chunk(2,dim=-1)
    u_pred = basis.idbt(hu).detach().to('cpu').numpy()
    v_pred = basis.idbt(hv).detach().to('cpu').numpy()
    u_error = time_rRMSE(u_gt, u_pred)
    v_error = time_rRMSE(v_gt, v_pred)
    extrapolated_u[model_name] = u_pred
    extrapolated_v[model_name] = v_pred
    extrapolated_error[model_name] = (u_error + v_error) / 2
    
    # Evaluating the extrapolation of baseline MLPs
    MLP_models = ("PINN", "QRes", "FLS")
    for model_name in MLP_models:
        model_path = os.path.join("results", "burgers2d", dir_name, model_name, f"seed_{seed}", "model.pt")
        if model_name == "PINN":
            model = PINN(in_dim=3, out_dim=2)
        elif model_name == "QRes":
            model = QRes(in_dim=3, out_dim=2)
        elif model_name == "FLS":
            model = FLS(in_dim=3, out_dim=2)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model = model.to(device)
        model.eval()
        uv = eval_chunked(model, x,y,t, n_chunks=10).to("cpu")
        u = uv[...,0].reshape((nt,nx,ny)).numpy()
        v = uv[...,1].reshape((nt,nx,ny)).numpy()
        u_error = time_rRMSE(u_gt, u)
        v_error = time_rRMSE(v_gt, v)
        extrapolated_u[model_name] = u
        extrapolated_v[model_name] = v
        extrapolated_error[model_name] = (u_error + v_error) / 2
    
    ts = ts.to("cpu")
    print("rRMSE for each model and time:")
    for model in ("PINN","QRes","FLS","NeuSA"):
        print(f"{model}:\t [0,1]: {np.mean(extrapolated_error[model][ts<=1.0]):.4f} \
              \t (1,2]: {np.mean(extrapolated_error[model][ts>1.0]):.4f}")
    
    save_dir = os.path.join("results","burgers2d","extrapolation",dir_name,f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    fig_u = plot_comparison(extrapolated_u, u_gt, extrapolated_error, time=ts, t_target = 2.)
    fig_v = plot_comparison(extrapolated_v, v_gt, extrapolated_error, time=ts, t_target = 2.)
    fig_u.savefig(os.path.join(save_dir, "u.png"), transparent=True, dpi=1200, bbox_inches="tight", format="png")
    fig_v.savefig(os.path.join(save_dir, "v.png"), transparent=True, dpi=1200, bbox_inches="tight", format="png")    
