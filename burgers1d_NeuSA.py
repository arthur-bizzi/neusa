import torch
from torchdyn.core import NeuralODE
import numpy as np
import time
import argparse
import os
import json
from math import pi
from scipy.interpolate import RegularGridInterpolator

from basis.fourier1d import FourierBasis1D
from models.hadamard_layer import HadamardLayer
from models.vector_fields import HeatLikeVectorField
from models.mlps import MLP
from utilities.trainer import AdamTrainer
from data_loaders.burgers1d_data_loader import Burgers1DGroundTruthLoader
from utilities.metrics import Metrics
from utilities.plot import create_time_evolution_gif


def loss_function(hu0,ts,NODE,basis,F,F0):
    hu = NODE(hu0,ts)
    Fhu = F(hu)

    u = basis.idbt(hu)
    u_x = basis.idbt(hu @ basis.D)
    
    u_t = basis.idbt(Fhu)
    nu_u_xx = basis.idbt(F0(hu))

    loss = torch.mean((u_t + u*u_x - nu_u_xx)**2)

    return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Burgers 1D')
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=1000)  
    parser.add_argument('--lr', type=float, default=0.01) 
    parser.add_argument('--freqs', type=int, default=201)
    
    args = parser.parse_args()
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    save_dir = os.path.join("results", "burgers1d", dir_name, f"{args.freqs}_freqs","NeuSA", f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    # domain, basis and grid
    domain_x = [-1.0,1.0]
    M = args.freqs
    basis = FourierBasis1D(M, domain_x)
    xs = basis.x
    
    domain_t = [0.0,1.0]
    time_steps = 2 * M - 1
    ts = torch.linspace(*domain_t,time_steps)
    
    # initial condition
    u0 = -torch.sin(pi*xs).squeeze(-1)
    hu0 = basis.dbt(u0)

    # initial vector field
    nu = 0.01/pi        # equation constant: u_t = -u*u_x + nu*u_xx
    D2 = basis.D2
    omegas = nu*torch.diag(D2)
    F0 = HadamardLayer(omegas)
    
    # neural networks and NODE
    in_dim = len(xs)
    hidden_dim = 4 * in_dim
    out_dim = in_dim
    num_hidden_layers = 2

    F_neural = MLP(in_dim,hidden_dim,out_dim,num_hidden_layers)
    F = HeatLikeVectorField(F_neural,F0,model_weight=0.1)
    NODE = NeuralODE(F,solver='rk4',sensitivity='autograd',return_t_eval=False)
    
    # training
    basis = basis.to(device)
    hu0 = hu0.to(device)
    ts = ts.to(device)
    NODE = NODE.to(device)
    F = F.to(device)
    F0 = F0.to(device)
    
    trainer = AdamTrainer(NODE.parameters(),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function= lambda: loss_function(hu0,ts,NODE,basis,F,F0))

    start_time = time.time()
    trainer.train_neusa()
    end_time = time.time()

    training_time = end_time - start_time

    # save trained model
    torch.save(F.state_dict(),os.path.join(save_dir, "model.pt"))

    # retrieve model solution
    hu = NODE(hu0,ts)
    u_pred = basis.idbt(hu).squeeze().detach().to('cpu').numpy()
    first_col = u_pred[:,[0]]
    u_pred = np.concatenate([u_pred,first_col],axis=1)

    # load ground-truth solution
    data_loader = Burgers1DGroundTruthLoader()
    u_gt = data_loader.get_ground_truth_solution()
    x_gt, t_gt = data_loader.get_ground_truth_grid()
    n_x_gt, n_t_gt = data_loader.get_grid_size()
    u_gt = u_gt.reshape(n_t_gt,n_x_gt)

    # interpolate predicted solution to ground-truth grid
    xs = xs.squeeze(-1).numpy()
    xs = np.concatenate([xs,[1]])
    ts = ts.cpu().numpy()

    interp = RegularGridInterpolator((ts,xs),u_pred)
    
    T,X = np.meshgrid(t_gt, x_gt, indexing='ij')
    points = np.stack([T.ravel(), X.ravel()], axis=-1)

    u_pred = interp(points).reshape(n_t_gt,n_x_gt)

    # compute metrics
    metrics= Metrics(u_pred,u_gt)

    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    metrics_dict = {
        "relative_l2_error": relative_l2_error_value,
        "relative_l1_error": relative_l1_error_value,
        "training_time": training_time
    }

    save_dir_metrics = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir_metrics,exist_ok=True)

    with open(os.path.join(save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    # #### PLOTS ####
    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    create_time_evolution_gif(x_gt,t_gt,u_gt,u_pred,figs_dir)