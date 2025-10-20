import argparse
import torch
import time
import os
import json

from models.PINN import PINN
from models.PINNsFormer import PINNsFormer
from models.QRes import QRes
from models.FLS import FLS

from utilities.network_initialization import init_weights
from utilities.grid import Regular2DGrid, Grid2DPINNsFormer
from utilities.trainer import AdamTrainer
from utilities.metrics import Metrics
from utilities.plot import create_graphs_grid,create_time_evolution_gif

from data_loaders.klein_gordon_data_loader import KleinGordon1DGroundTruthLoader

def loss_function(model, training_grid, sigma, sqrt_2pi, m, device):
    training_grid = training_grid.to(device)

    x = training_grid.x      
    t0 = training_grid.t0.requires_grad_()      
        
    t = training_grid.t       
    x_lb = training_grid.x_lb  
    x_ub = training_grid.x_ub 

    x_res = training_grid.grid_column_x.requires_grad_()
    t_res = training_grid.grid_column_t.requires_grad_()
    
    u0 = torch.exp(-(x**2)/(2*sigma*sigma)) / (sqrt_2pi * sigma)

    U0 = model(x,t0)
    U0_t = torch.autograd.grad(U0, t0, grad_outputs=torch.ones_like(U0), create_graph=True)[0]

    loss_ic = 1e3*(torch.mean((U0-u0)**2) + torch.mean(U0_t**2))
    loss_bc = torch.mean((model(x_lb,t))**2) + torch.mean((model(x_ub,t))**2)

    u = model(x_res,t_res)
    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(u_t), create_graph = True)[0]

    loss_res = torch.mean((u_tt - u_xx  + m*torch.sin(u))**2)

    return (loss_ic, loss_bc, loss_res)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Klein-Gordon baseline models')
    parser.add_argument('--model', type=str, default='PINN')    # options: 'PINN', 'FLS', 'QRes', 'PINNsFormer
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=10000)  
    parser.add_argument('--lr', type=float, default=0.001)  
      
    args = parser.parse_args()
    model_name = args.model
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr

    torch.manual_seed(seed)

    save_dir = os.path.join("results", "klein_gordon", dir_name, model_name, f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    # initialize model
    if model_name == "PINN":
        model = PINN(in_dim=2, out_dim=1)
    elif model_name == "QRes":
        model = QRes(in_dim=2, out_dim=1)
    elif model_name == "PINNsFormer":
        model = PINNsFormer(in_dim=2, out_dim=1)
    elif model_name == "FLS":
        model = FLS(in_dim=2, out_dim=1)
    model.apply(init_weights)
    model = model.to(device)

    # domain and grid
    domain_x = [-4.0,4.0]
    domain_t = [0.0,3.0]
    if model_name == "PINNsFormer":
        n_x = 101
        n_t = 101
        training_grid = Grid2DPINNsFormer(domain_x, domain_t, n_x,n_t)
    else:
        n_x = 201
        n_t = 201
        training_grid = Regular2DGrid(domain_x, domain_t, n_x,n_t)

    # constants
    sigma = 0.1
    sigma = torch.tensor(sigma,dtype=torch.float32).to(device)
    sqrt_2pi = torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=torch.float32)).to(device)
    m = torch.tensor(10,dtype=torch.float32).to(device)
    
    # training
    trainer = AdamTrainer(params=list(model.parameters()),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function=lambda: loss_function(model, training_grid, sigma, sqrt_2pi, m, device)
                )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time = end_time - start_time

    # save trained model
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # load ground-truth solution
    data_loader = KleinGordon1DGroundTruthLoader(model_name)
    x_gt,t_gt = data_loader.get_ground_truth_grid()
    u_gt = data_loader.get_ground_truth_solution()

    n_t_gt, n_x_gt = data_loader.get_grid_size()

    x,t = data_loader.get_ground_truth_grid_as_tensors()

    # retrieve model solution
    if model_name == "PINNsFormer":
        with torch.no_grad():
            x = x.to(device)
            t = t.to(device)
            
            start_time = time.time()
            u_pred = model(x,t)
            u_pred = u_pred[:,0,:]
            end_time = time.time()
            inference_time = end_time-start_time
                
            u_pred = u_pred.squeeze().cpu().numpy()

    else:
        with torch.no_grad():
            
            x = x.to(device)
            t = t.to(device)
            
            start_time = time.time()
            u_pred = model(x,t)
            end_time = time.time()
            inference_time = end_time-start_time

            u_pred = u_pred.squeeze().cpu().numpy()
            
    # compute metrics
    metrics = Metrics(u_pred,u_gt)
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

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)
    
    u_gt = u_gt.reshape(n_t_gt,n_x_gt)
    u_pred = u_pred.reshape(n_t_gt,n_x_gt)

    offset = n_t_gt // 3
    u_plot = [u_gt[0,:], u_gt[offset,:], u_gt[2 * offset,:], u_gt[n_t_gt-1,:]]
    u_pred_plot = [u_pred[0,:], u_pred[offset,:], u_pred[2 * offset,:], u_pred[n_t_gt-1,:]]

    x_plot = x_gt
    titles = [f"t={t_gt[0].item()}", f"t={t_gt[offset].item()}", f"t={t_gt[2 * offset].item()}", f"t={t_gt[n_t_gt-1].item()}"]

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    create_graphs_grid(x_plot,u_plot,u_pred_plot,titles,figs_dir)
    create_time_evolution_gif(x_gt,t_gt,u_gt,u_pred,figs_dir)
    