import argparse
import torch
import time
import os
import numpy as np
import json

from models.PINN import PINN
from models.QRes import QRes
from models.FLS import FLS

from utilities.network_initialization import init_weights
from utilities.grid import Random3DGrid
from utilities.trainer import AdamTrainer
from utilities.metrics import Metrics
from utilities.plot import plot_prediction_vs_groundtruth
from scipy.interpolate import RegularGridInterpolator


def loss_function(model, training_grid, sigma, center, c_squared_interp, device):
    training_grid.update_random_grid()
    training_grid.to(device)

    x_ic = training_grid.x_ic
    y_ic = training_grid.y_ic
    t_ic = training_grid.t_ic.detach().requires_grad_()

    x_res = training_grid.x_res.detach().requires_grad_()
    y_res = training_grid.y_res.detach().requires_grad_()
    t_res = training_grid.t_res.detach().requires_grad_()
    
    u0 = torch.exp(-((x_ic-center[0])**2+(y_ic-center[1])**2)/(2*sigma**2))

    u_ic = model(x_ic,y_ic,t_ic)
    u_t_ic = torch.autograd.grad(u_ic, t_ic, grad_outputs=torch.ones_like(u_ic), create_graph=True)[0]

    loss_ic = torch.mean((u_ic-u0)**2) + torch.mean((u_t_ic)**2)
    
    loss_bc = torch.tensor(0.0)

    u = model(x_res,y_res,t_res)
    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_res, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_t = torch.autograd.grad(u,t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]

    x_res = x_res.detach().cpu().numpy()
    y_res = y_res.detach().cpu().numpy()
    points = np.stack([x_res,y_res],axis=-1)

    c_squared = c_squared_interp(points)

    c_squared = torch.tensor(c_squared).to(device)

    loss_res = torch.mean((u_tt - c_squared*(u_xx+u_yy))**2)

    return (1e3*loss_ic, loss_bc, loss_res)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Wave 2d Marmousi baseline models')
    parser.add_argument('--model', type=str, default='PINN')    # options: 'PINN', 'FLS', 'QRes', 'PINNsFormer
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')     
    parser.add_argument('--steps', type=int, default=20000)  
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    model_name = args.model
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr

    torch.manual_seed(seed)

    save_dir = os.path.join("results", "wave2d_marmousi", dir_name, model_name, f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    # initialize model
    if model_name == "PINN":
        model = PINN(in_dim=3, out_dim=1)
    elif model_name == "QRes":
        model = QRes(in_dim=3, out_dim=1)
    elif model_name == "FLS":
        model = FLS(in_dim=3, out_dim=1)
    model.apply(init_weights)
    model = model.to(device)

    # domain and grid
    domain_x = [-2.0,2.0]
    domain_y = [-2.0,2.0]
    domain_t = [0.0,2.0]
    
    n_ic = 1000
    n_bc = 500
    n_res = 10000
    training_grid = Random3DGrid(domain_x, domain_y, domain_t, n_ic, n_bc, n_res)

    # initial condition parameters
    sigma = torch.tensor(0.1,dtype=torch.float32).to(device)
    center = (0.0,-0.5)

    # load velocity profile (and interpolate to the training grid)
    velocity_profile = np.loadtxt("data/wave2d_marmousi/velocity_profile.txt",dtype=np.float32)  # 401 x 401
    c_squared = torch.tensor(velocity_profile * velocity_profile)
    c_squared = c_squared[100:301,100:301].numpy()
    xs = np.linspace(*domain_x,201)
    ys = np.linspace(*domain_y,201)
    c_squared_interp = RegularGridInterpolator((xs,ys),c_squared) 
    
    # training
    trainer = AdamTrainer(params=list(model.parameters()),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function=lambda: loss_function(model, training_grid, sigma,center,c_squared_interp, device)
                )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time = end_time - start_time

    # save trained model
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # load ground-truth solution
    x_gt = torch.linspace(*domain_x,201)
    y_gt = torch.linspace(*domain_y,201)
    t_gt = torch.linspace(*domain_t,201)
    u_gt = np.load("data/wave2d_marmousi/u_gt.npy")

    # retrieve model solution    
    n_x_gt, n_y_gt, n_t_gt = (201,201,201)

    T,X,Y = torch.meshgrid(t_gt,x_gt,y_gt,indexing='ij')
    x = X.flatten().unsqueeze(-1)
    y = Y.flatten().unsqueeze(-1)
    t = T.flatten().unsqueeze(-1)
    
    chunk_size = 10000  
    num_points = x.shape[0]

    u_pred_list = []
    with torch.no_grad():
        x = x.to(device)
        y = y.to(device)
        t = t.to(device)

        start_time = time.time()
        for i in range(0, num_points, chunk_size):
            x_chunk = x[i:i+chunk_size]
            y_chunk = y[i:i+chunk_size]
            t_chunk = t[i:i+chunk_size]
            
            u_chunk = model(x_chunk, y_chunk, t_chunk)
            u_chunk = u_chunk.squeeze().cpu().numpy()
            u_pred_list.append(u_chunk)

    u_pred = np.concatenate(u_pred_list, axis=0)
    end_time = time.time()
    inference_time = end_time-start_time
    
    # compute metrics
    u_gt = u_gt.reshape(n_t_gt*n_x_gt*n_y_gt)

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

    u_pred = u_pred.reshape(n_t_gt,n_x_gt,n_y_gt)
    u_gt = u_gt.reshape(n_t_gt,n_x_gt,n_y_gt)

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    save_path = os.path.join(figs_dir,"plots.png")
    plot_prediction_vs_groundtruth(ts=t_gt,u_pred=u_pred,u_gt=u_gt,extent=[*domain_x,*domain_y],save_path=save_path)