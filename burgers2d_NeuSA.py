import torch
from torchdyn.core import NeuralODE
import numpy as np
import time
import json
import argparse
import os

from basis.fourier2d import FourierBasis2D
from models.mlps import MLPWithTranspose2DWith2DOutput
from models.vector_fields import HeatLikeVectorFieldWith2DOutput
from models.hadamard_layer import HadamardLayer
from utilities.trainer import AdamTrainer
from data_loaders.burgers2d_data_loader import Burgers2DGroundTruthLoader
from utilities.metrics import Metrics
from utilities.plot import plot_prediction_vs_groundtruth

def loss_function(huv0,basis,NODE,F,F0,ts):
    huv = NODE(huv0,ts)
    Fhuv = F(huv)

    hu, hv = huv.chunk(2,dim=-1)
    Fhu,Fhv = Fhuv.chunk(2,dim=-1)

    u = basis.idbt(hu)
    v = basis.idbt(hv)

    hu_x = torch.einsum('ik,...kj->...ij',basis.Dx,hu)
    hu_y = torch.einsum('...ik,kj->...ij',hu,basis.Dy)

    hv_x = torch.einsum('ik,...kj->...ij',basis.Dx,hv)
    hv_y = torch.einsum('...ik,kj->...ij',hv,basis.Dy)

    u_x = basis.idbt(hu_x)
    u_y = basis.idbt(hu_y)

    v_x = basis.idbt(hv_x)
    v_y = basis.idbt(hv_y)

    right_hand_side_u = basis.idbt(F0(hu)) - u*u_x - v*u_y
    u_t= basis.idbt(Fhu)

    right_hand_side_v = basis.idbt(F0(hv)) -u*v_x - v*v_y
    v_t= basis.idbt(Fhv)
    
    loss_u = torch.mean((u_t - right_hand_side_u)**2)
    loss_v = torch.mean((v_t - right_hand_side_v)**2)
    loss = loss_u + loss_v

    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Burgers 2d NeuSA')
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=200)  
    parser.add_argument('--lr', type=float, default=0.005) 

    args = parser.parse_args()
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr

    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    save_dir = os.path.join("results", "burgers2d", dir_name, "NeuSA", f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    # domain and resolution
    domain_x = [0.0,4.0]
    domain_y = [0.0,4.0]
    domain_t = [0.0,1.0]

    M = 201
    N = 201
    time_steps = 201

    ts = torch.linspace(*domain_t,time_steps)

    # basis and grid
    basis = FourierBasis2D(M,N,domain_x,domain_y,device=device)
    X,Y = torch.meshgrid(basis.x,basis.y,indexing='ij')
    grid = torch.stack((X,Y),dim=-1)    # shape = (M,N,2)
    xs = grid[:,:,0:1]
    ys = grid[:,:,1:2]

    # initial condition
    u0 = (torch.sin(np.pi*xs)*torch.sin(np.pi*ys)).squeeze(-1).to(device)    # shape = (M,N)
    v0 = (torch.cos(np.pi*xs)*torch.cos(np.pi*ys)).squeeze(-1).to(device)    # shape = (M,N)
    hu0 = basis.dbt(u0)
    hv0 = basis.dbt(v0)
    huv0 = torch.cat([hu0,hv0],dim=-1)  # shape = (M,2N)
    
    # initial vector field
    omegas = 0.01*basis.D2
    F0 = HadamardLayer(omegas)

    # neural networks and NODEs
    F_neural_u = MLPWithTranspose2DWith2DOutput(M=M,N=N)
    F_neural_v = MLPWithTranspose2DWith2DOutput(M=M,N=N)
    F = HeatLikeVectorFieldWith2DOutput(F_neural_u,F_neural_v,F0,model_weight=0.1)
    NODE = NeuralODE(F,solver='rk4',sensitivity='autograd',return_t_eval=False)

    # training
    huv0 = huv0.to(device)
    ts = ts.to(device)
    basis = basis.to(device)
    NODE = NODE.to(device)
    F0 = F0.to(device)
    F = F.to(device)

    trainer = AdamTrainer(NODE.parameters(),
                adam_lr=lr,
                adam_steps=steps,
                learning_rate_scheduler=True,
                loss_function= lambda: loss_function(huv0,basis,NODE,F,F0,ts)
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

    hu,hv = huv.chunk(2,dim=-1)
    u_pred = basis.idbt(hu).detach().to('cpu').numpy()
    time_idx = range(0,time_steps,2)
    u_pred = u_pred[time_idx,:,:]
    v_pred = basis.idbt(hv).detach().to('cpu').numpy()
    v_pred = v_pred[time_idx,:,:]
    ts = ts[time_idx]

    # load ground-truth solution
    data_loader = Burgers2DGroundTruthLoader()
    u_gt = data_loader.get_ground_truth_solution_u().reshape(len(time_idx),M,N)
    v_gt = data_loader.get_ground_truth_solution_v().reshape(len(time_idx),M,N)

    # compute metrics
    metrics_u = Metrics(u_pred,u_gt)
    metrics_v = Metrics(v_pred,v_gt)
    relative_l2_error_value = (metrics_u.relative_l2_error()+metrics_v.relative_l2_error())/2
    relative_l1_error_value = (metrics_u.relative_l1_error()+metrics_v.relative_l1_error())/2

    metrics_dict = {
        "relative_l2_error": float(relative_l2_error_value),
        "relative_l1_error": float(relative_l1_error_value),
        "training_time": float(training_time),
        "inference_time": float(inference_time),
    }

    save_dir_metrics = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir_metrics,exist_ok=True)

    with open(os.path.join(save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    # #### PLOTS ####

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    save_path = os.path.join(figs_dir,"plots_u.png")
    plot_prediction_vs_groundtruth(ts=ts,u_pred=u_pred,u_gt=u_gt,extent=[*domain_x,*domain_y],save_path=save_path)
    
    save_path = os.path.join(figs_dir,"plots_v.png")
    plot_prediction_vs_groundtruth(ts=ts,u_pred=v_pred,u_gt=v_gt,extent=[*domain_x,*domain_y],save_path=save_path)