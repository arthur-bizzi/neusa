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
from utilities.grid import Random4DGrid
from utilities.trainer import AdamTrainer
from utilities.metrics import Metrics
from utilities.plot import generate_figures_3d

from data_loaders.wave3d_data_loader import wave3d_groundtruth_generator

def velocity_profile(z, a=1.0, b=-0.5, c=1000, d=0.5):
        return a + b * (torch.sigmoid(c * (z - d)))

def neumann_loss(model, x, y, z, t, axis):
    if axis == "x":
        x.requires_grad_()
        u = model(x, y, z, t)
        u_n = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    elif axis == "y":
        y.requires_grad_()
        u = model(x, y, z, t)
        u_n = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    elif axis == "z":
        z.requires_grad_()
        u = model(x, y, z, t)
        u_n = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return torch.mean(u_n**2)

def loss_function(model, training_grid, sigma, device):
    training_grid.update_random_grid()
    training_grid.to(device)

    x_ic = training_grid.x_ic
    y_ic = training_grid.y_ic
    z_ic = training_grid.z_ic
    t_ic = training_grid.t_ic.detach().requires_grad_()

    x_res = training_grid.x_res.detach().requires_grad_()
    y_res = training_grid.y_res.detach().requires_grad_()
    z_res = training_grid.z_res.detach().requires_grad_()
    t_res = training_grid.t_res.detach().requires_grad_()
    
    u0 = 10*torch.exp(-(x_ic**2+y_ic**2+z_ic**2)/(2*sigma**2))

    u_ic = model(x_ic,y_ic,z_ic,t_ic)
    u_t_ic = torch.autograd.grad(u_ic, t_ic, grad_outputs=torch.ones_like(u_ic), create_graph=True)[0]

    loss_ic = torch.mean((u_ic-u0)**2) + torch.mean((u_t_ic)**2)
    
    bc_x0_loss = neumann_loss(model, training_grid.x_bc_x0, training_grid.y_bc_x0, training_grid.z_bc_x0, training_grid.t_bc_x0, axis="x")
    bc_x1_loss = neumann_loss(model, training_grid.x_bc_x1, training_grid.y_bc_x1, training_grid.z_bc_x1, training_grid.t_bc_x1, axis="x")
    bc_y0_loss = neumann_loss(model, training_grid.x_bc_y0, training_grid.y_bc_y0, training_grid.z_bc_y0, training_grid.t_bc_y0, axis="y")
    bc_y1_loss = neumann_loss(model, training_grid.x_bc_y1, training_grid.y_bc_y1, training_grid.z_bc_y1, training_grid.t_bc_y1, axis="y")
    bc_z0_loss = neumann_loss(model, training_grid.x_bc_z0, training_grid.y_bc_z0, training_grid.z_bc_z0, training_grid.t_bc_z0, axis="z")
    bc_z1_loss = neumann_loss(model, training_grid.x_bc_z1, training_grid.y_bc_z1, training_grid.z_bc_z1, training_grid.t_bc_z1, axis="z")
    loss_bc = (bc_x0_loss + bc_x1_loss + bc_y0_loss + bc_y1_loss + bc_z0_loss + bc_z1_loss) / 6.0

    c = velocity_profile(z_res)

    u = model(x_res,y_res,z_res,t_res)
    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_res, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z_res, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z,z_res,grad_outputs=torch.ones_like(u_z),create_graph=True)[0]
    u_t = torch.autograd.grad(u,t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]

    loss_res = torch.mean((u_tt - (c**2)*(u_xx+u_yy+u_zz))**2)

    return (1e3*loss_ic, loss_bc, loss_res)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Wave 3d baseline models')
    parser.add_argument('--model', type=str, default='PINN')    # options: 'PINN', 'FLS', 'QRes'
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

    save_dir = os.path.join("results", "wave3d", dir_name, model_name, f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    # initialize model
    if model_name == "PINN":
        model = PINN(in_dim=4, out_dim=1)
    elif model_name == "QRes":
        model = QRes(in_dim=4, out_dim=1)
    elif model_name == "FLS":
        model = FLS(in_dim=4, out_dim=1)
    model.apply(init_weights)
    model = model.to(device)

    # domain and grid
    domain_x = [-1.5,1.5]
    domain_y = [-1.5,1.5]
    domain_z = [-1.5,1.5]
    domain_t = [0.0,1.0]
    
    n_ic = 10000
    n_bc = 2000
    n_res = 70000

    training_grid = Random4DGrid(domain_x, domain_y, domain_z, domain_t, n_ic, n_res, n_bc)
    
    # constants
    sigma = torch.tensor(0.15,dtype=torch.float32).to(device)

    # training
    trainer = AdamTrainer(params=list(model.parameters()),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function=lambda: loss_function(model, training_grid, sigma, device)
                )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time = end_time - start_time

    # save model
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # retrieve model solution
    xs = torch.linspace(*domain_x,101)
    ys = torch.linspace(*domain_y,101)
    zs = torch.linspace(*domain_z,101)
    ts = torch.linspace(*domain_t,111)

    T,X,Y,Z = torch.meshgrid([ts,xs,ys,zs], indexing='ij')
    x = X.flatten().unsqueeze(-1)
    y = Y.flatten().unsqueeze(-1)
    z = Z.flatten().unsqueeze(-1)
    t = T.flatten().unsqueeze(-1)
    
    chunk_size = 10000
    num_points = x.shape[0]
    u_pred_list = []

    
    with torch.no_grad():
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        t = t.to(device)
        
        start_time = time.time()
        for i in range(0,num_points,chunk_size):
            x_chunk = x[i:i+chunk_size]
            y_chunk = y[i:i+chunk_size]
            z_chunk = z[i:i+chunk_size]
            t_chunk = t[i:i+chunk_size]           
            
            u_pred_chunk = model(x_chunk,y_chunk,z_chunk,t_chunk)
            u_pred_list.append(u_pred_chunk)

        u_pred = torch.cat(u_pred_list,dim=0)
        
        end_time = time.time()
        inference_time = end_time-start_time

        u_pred = u_pred.squeeze()
        u_pred = u_pred.reshape(len(ts),len(xs),len(ys),len(zs))
        u_pred = u_pred.cpu().numpy()
    
    np.save(f"u_pred_{model_name}.npy",u_pred)
    # load ground-truth solution
    u_gt = wave3d_groundtruth_generator()

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

    xs = np.linspace(*domain_x,101)
    ys = np.linspace(*domain_y,101)
    zs = np.linspace(*domain_z,101)
    ts = np.linspace(*domain_t,111)

    u_gt = np.expand_dims(u_gt,axis=-1)
    u_pred = np.expand_dims(u_pred,axis=-1)

    generate_figures_3d(ts,xs,ys,zs,u_pred,u_gt,save_dir_figs=figs_dir,order="txyz",model_name=model_name)
   