import torch
from torchdyn.core import NeuralODE
import time
import argparse
import os
import json

from basis.sine1d import SineBasis1D
from models.hadamard_layer import HadamardLayer
from models.mlps import MLP
from models.vector_fields import WaveLikeVectorField
from utilities.trainer import AdamTrainer
from data_loaders.klein_gordon_triangular_data_loader import KleinGordonTriangular1DGroundTruthLoader
from utilities.metrics import Metrics
from utilities.plot import create_graphs_grid, create_time_evolution_gif

def triangle(x: torch.Tensor, width = 1., center = (0.,0.)):
    u = x*0.
    x0 = center[0] - width
    x1 = center[0]
    x2 = center[0] + width
    rising_mask = torch.logical_and(x>=x0, x<x1)
    descending_mask = torch.logical_and(x>=x1, x<x2)
    u[rising_mask] = (x[rising_mask] - x0) / width
    u[descending_mask] = 1 - (x[descending_mask]-x1) / width

    return u

def loss_function(huv0,ts,NODE,basis,F,F0):
    huv = NODE(huv0,ts)
    Fhuv = F(huv)

    hu,_ = huv.chunk(2,dim=-1)
    _,Fhv = Fhuv.chunk(2,dim=-1)

    # u_tt = v_t = D^2(u) - 10sin(u)
    right_hand_side = basis.idbt(F0(hu)) - 10 * torch.sin(basis.idbt(hu))
    v_t = basis.idbt(Fhv)

    loss = torch.mean((v_t - right_hand_side)**2)

    return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Klein-Gordon Triangular NeuSA')
    parser.add_argument('--dir', type=str, default='free_experiment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=1000)  
    parser.add_argument('--lr', type=float, default=0.02)   

    args = parser.parse_args()
    dir_name = args.dir
    seed = args.seed
    device = args.device
    seed = args.seed
    steps = args.steps
    lr = args.lr

    torch.set_float32_matmul_precision('high')
    torch.manual_seed(seed)

    save_dir = os.path.join("results", "klein_gordon_triangular", dir_name, "NeuSA", f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    # domain, basis and grid
    domain_x = [-4.0,4.0]
    M = 201
    basis = SineBasis1D(M,domain_x)
    xs = basis.x  

    domain_t = [0.0,3.0]
    time_steps = 301
    ts = torch.linspace(*domain_t,time_steps)

    # initial condition
    u0 = triangle(xs)
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

    F = F.to(device)
    NODE = NODE.to(device)
    
    # training
    basis = basis.to(device)
    huv0 = huv0.to(device)
    ts = ts.to(device)
    F0 = F0.to(device)

    trainer = AdamTrainer(NODE.parameters(),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function= lambda: loss_function(huv0,ts,NODE,basis,F,F0)
                )
        
    start_time = time.time()
    
    trainer.train_neusa()

    end_time = time.time()
    training_time = end_time - start_time

    # save trained model
    torch.save(F.state_dict(),os.path.join(save_dir, "model.pt"))

    # retrieve model solution
    start_time = time.time()
    huv = NODE(huv0,ts)
    end_time = time.time()
    inference_time = end_time-start_time
    hu,_ = huv.chunk(2,dim=-1)
    u_pred = basis.idbt(hu).detach().to('cpu').numpy()


    # load ground-truth
    data_loader = KleinGordonTriangular1DGroundTruthLoader()
    u_gt = data_loader.get_ground_truth_solution()
    x_gt, t_gt = data_loader.get_ground_truth_grid()
    n_t_gt,n_x_gt = data_loader.get_grid_size()

    u_gt = u_gt.reshape(n_t_gt,n_x_gt)
    
    # compute and save metrics
    metrics= Metrics(u_pred,u_gt)

    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    metrics_dict = {
        "relative_l2_error": float(relative_l2_error_value),
        "relative_l1_error": float(relative_l1_error_value),
        "training_time": float(training_time),
        "inference_time": float(inference_time)
    }

    save_dir_metrics = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir_metrics,exist_ok=True)

    with open(os.path.join(save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
    

    # #### PLOTS ####

    offset = n_t_gt // 3
    u_plot = [u_gt[0,:], u_gt[offset,:], u_gt[2 * offset,:], u_gt[n_t_gt-1,:]]
    u_pred_plot = [u_pred[0,:], u_pred[offset,:], u_pred[2 * offset,:], u_pred[n_t_gt-1,:]]

    x_plot = x_gt
    titles = [f"t={t_gt[0].item()}", f"t={t_gt[offset].item()}", f"t={t_gt[2 * offset].item()}", f"t={t_gt[n_t_gt-1].item()}"]

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    create_graphs_grid(x_plot,u_plot,u_pred_plot,titles,figs_dir)
    create_time_evolution_gif(x_gt,t_gt,u_gt,u_pred,figs_dir)
