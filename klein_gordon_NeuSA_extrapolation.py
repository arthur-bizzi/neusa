import torch
from torchdyn.core import NeuralODE
import numpy as np
import time
import argparse
import os
from math import pi, sqrt

from basis.sine1d import SineBasis1D
from models.hadamard_layer import HadamardLayer
from models.mlps import MLP
from models.vector_fields import WaveLikeVectorField
from utilities.trainer import AdamTrainer
from data_loaders.klein_gordon_data_loader import KleinGordon1DGroundTruthLoader
from utilities.metrics import Metrics
from utilities.plot import plot_extrapolation_rl2, plot_extrapolation_rl1

def loss_function(huv0,ts,NODE,basis,F,F0):
    huv = NODE(huv0,ts)
    Fhuv = F(huv)

    hu,_ = huv.chunk(2,dim=-1)
    _,Fhv = Fhuv.chunk(2,dim=-1)


    right_hand_side = basis.idbt(F0(hu)) - 10*torch.sin(basis.idbt(hu))
    v_t = basis.idbt(Fhv)

    loss = torch.mean((v_t-right_hand_side)**2)

    return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Klein-Gordon NeuSA time extrapolation')
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=1000)  
    parser.add_argument('--lr', type=float, default=0.02) 
    args = parser.parse_args()
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    save_dir = os.path.join("results", "klein_gordon_extrapolation", dir_name, "NeuSA", f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    # domain, basis and grid
    domain_x = [-4.0,4.0]
    M = 201
    basis = SineBasis1D(M,domain_x)
    xs = basis.x  

    # time domains
    domain_t = [0.0,3.0]
    time_steps = 201
    ts = torch.linspace(*domain_t,time_steps)

    # we train for half the time domain
    ts_ext = ts[:int(len(ts)/2)]

    # initial condition
    sigma = 0.1
    u0 = torch.exp(-(xs**2)/(2*sigma*sigma))/ (sqrt(2*pi)*sigma)
    v0 = torch.zeros_like(u0)
    hu0 = basis.dbt(u0)
    hv0 = basis.dbt(v0)
    huv0 = torch.cat([hu0,hv0],dim=-1)
   
    # initial vector field
    D2 = basis.D2
    omegas = torch.diag(D2)
    F0 = HadamardLayer(omegas)
    
    # neural networks and NODE
    in_dim = len(xs)
    hidden_dim = 4 * in_dim
    out_dim = in_dim
    num_hidden_layers = 2

    F_neural = MLP(in_dim,hidden_dim,out_dim,num_hidden_layers)
    F = WaveLikeVectorField(F_neural,F0,model_weight=0.1)
    NODE = NeuralODE(F,solver='rk4',sensitivity='autograd',return_t_eval=False)

    # training
    basis = basis.to(device)
    huv0 = huv0.to(device)
    ts = ts.to(device)
    ts_ext = ts_ext.to(device)
    NODE = NODE.to(device)
    F = F.to(device)
    F0 = F0.to(device)
    
    trainer = AdamTrainer(NODE.parameters(),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function= lambda: loss_function(huv0,ts_ext,NODE,basis,F,F0)
                )

    start_time = time.time()
    trainer.train_neusa()
    end_time = time.time()

    training_time = end_time - start_time

    # save trained model
    torch.save(F.state_dict(),os.path.join(save_dir, "model.pt"))

    # retrieve model solution
    huv = NODE(huv0,ts)
    hu,_ = huv.chunk(2,dim=-1)
    u_pred = basis.idbt(hu).squeeze().detach().to('cpu').numpy()

    # load ground-truth solution
    data_loader = KleinGordon1DGroundTruthLoader()
    u_gt = data_loader.get_ground_truth_solution()
    x_gt, t_gt = data_loader.get_ground_truth_grid()
    n_x_gt, n_t_gt = data_loader.get_grid_size()

    u_gt = u_gt.reshape(n_t_gt,n_x_gt)
    
    # compute rMSE's by time
    metrics= Metrics(u_pred,u_gt)

    save_dir_metrics = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir_metrics,exist_ok=True)

    relative_l2_error_value_ext = metrics.rl2_by_time()
    relative_l1_error_value_ext = metrics.rl1_by_time()

    np.savetxt(os.path.join(save_dir_metrics, "times.txt"), t_gt)
    np.savetxt(os.path.join(save_dir_metrics, "rl2s.txt"), relative_l2_error_value_ext)

    # #### PLOTS ####
    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    save_dir_metrics_rl2s = os.path.join(figs_dir, "rl2s.png")
    save_dir_metrics_rl1s = os.path.join(figs_dir, "rl1s.png")

    plot_extrapolation_rl2(ts.cpu().numpy(),relative_l2_error_value_ext,save_dir_metrics_rl2s)
    plot_extrapolation_rl1(ts.cpu().numpy(),relative_l1_error_value_ext,save_dir_metrics_rl1s)





    
   