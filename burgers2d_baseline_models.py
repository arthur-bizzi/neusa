import argparse
import torch
import time
import numpy as np
import os
import json

from models.PINN import PINN
from models.PINNsFormer import PINNsFormer
from models.QRes import QRes
from models.FLS import FLS

from utilities.network_initialization import init_weights
from utilities.grid import Random3DGrid, Random3DGridPINNsFormer
from utilities.trainer import AdamTrainer
from utilities.metrics import Metrics
from data_loaders.burgers2d_data_loader import Burgers2DGroundTruthLoader
from utilities.plot import plot_prediction_vs_groundtruth


def loss_function(model, training_grid, nu, pi, device):
    training_grid.update_random_grid()
    training_grid.to(device)

    x_ic = training_grid.x_ic
    y_ic = training_grid.y_ic
    t_ic = training_grid.t_ic

    x_lb_bc = training_grid.x_lb_bc
    x_ub_bc = training_grid.x_ub_bc
    x_bc = training_grid.x_bc
    y_lb_bc = training_grid.y_lb_bc
    y_ub_bc = training_grid.y_ub_bc
    y_bc = training_grid.y_bc
    t_bc = training_grid.t_bc

    x_res = training_grid.x_res.detach().requires_grad_()
    y_res = training_grid.y_res.detach().requires_grad_()
    t_res = training_grid.t_res.detach().requires_grad_()

    
    u0 = torch.sin(pi*x_ic)*torch.sin(pi*y_ic),torch.cos(pi*x_ic)*torch.cos(pi*y_ic)
    u0 = torch.cat(u0,dim=-1)

    loss_ic = torch.mean((model(x_ic,y_ic,t_ic)-u0)**2)
    
    loss_bc = torch.mean((model(x_lb_bc,y_bc,t_bc)-model(x_ub_bc,y_bc,t_bc))**2) + \
                torch.mean((model(x_bc,y_lb_bc,t_bc)-model(x_bc,y_ub_bc,t_bc))**2)

    output = model(x_res,y_res,t_res)
    u = output[:,0:1]
    v = output[:,1:2]

    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_res, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x_res, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x_res, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y_res, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y_res, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t_res, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    loss_res = torch.mean((u_t + u*u_x + v*u_y + nu*(u_xx + u_yy))**2) + \
        torch.mean((v_t + u*v_x + v*v_y + nu*(v_xx + v_yy))**2)

    return (1e2*loss_ic, loss_bc, loss_res)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Burgers 2d baseline models')
    parser.add_argument('--model', type=str, default='PINN')    # options: 'PINN', 'FLS', 'QRes', 'PINNsFormer
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default=42)    # random seed
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

    save_dir = os.path.join("results", "burgers2d", dir_name, model_name, f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    # initialize model
    if model_name == "PINN":
        model = PINN(in_dim=3, out_dim=2)
    elif model_name == "PINNsFormer":
        model = PINNsFormer(in_dim=3, out_dim=2)
    elif model_name == "QRes":
        model = QRes(in_dim=3, out_dim=2)
    elif model_name == "FLS":
        model = FLS(in_dim=3, out_dim=2)
    model.apply(init_weights)
    model = model.to(device)
    
    # domain and grid
    domain_x = [0.0,4.0]
    domain_y = [0.0,4.0]
    domain_t = [0.0,1.0]
    if model_name == "PINNsFormer":
        n_ic = 200
        n_bc = 100
        n_res = 2000
        training_grid = Random3DGridPINNsFormer(domain_x, domain_y, domain_t, n_ic, n_bc, n_res)
    else:
        n_ic = 1000
        n_bc = 500
        n_res = 10000
        training_grid = Random3DGrid(domain_x, domain_y, domain_t, n_ic, n_bc, n_res)

    # constants
    nu = torch.tensor(-0.01,dtype=torch.float32, device=device)
    pi = torch.tensor(np.pi, dtype=torch.float32, device=device)

    # training
    trainer = AdamTrainer(params=list(model.parameters()),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function=lambda: loss_function(model, training_grid, nu, pi, device)
                    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time = end_time - start_time
    
    # save trained model
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # load ground-truth solution
    data_loader = Burgers2DGroundTruthLoader(model_name)
    x_gt,y_gt,t_gt = data_loader.get_ground_truth_grid()
    u_gt = data_loader.get_ground_truth_solution_u()
    v_gt = data_loader.get_ground_truth_solution_v()

    n_x_gt, n_y_gt, n_t_gt = data_loader.get_grid_size()

    x,y,t = data_loader.get_ground_truth_grid_as_tensors()

    # retrieve model solution
    if model_name == "PINNsFormer":
        chunk_size = 1000  
        num_points = x.shape[0]
        
        x = x.to(device)
        y = y.to(device)
        t = t.to(device)
        
        u_pred_list = []
        v_pred_list = []
        start_time = time.time()
        with torch.no_grad():
            for i in range(0, num_points, chunk_size):
                x_chunk = x[i:i+chunk_size]
                y_chunk = y[i:i+chunk_size]
                t_chunk = t[i:i+chunk_size]
                
                uv_chunk = model(x_chunk, y_chunk, t_chunk)[:, 0, :]
                uv_chunk = uv_chunk.squeeze().cpu().numpy()

                u_pred_list.append(uv_chunk[:, 0])
                v_pred_list.append(uv_chunk[:, 1])
            
            u_pred = np.concatenate(u_pred_list,axis=0)
            v_pred = np.concatenate(v_pred_list, axis=0)
            
            end_time = time.time()
            inference_time = end_time - start_time
        
    else:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            t = t.to(device)
            
            start_time = time.time()
            uv_pred = model(x,y,t)
            end_time = time.time()
            inference_time = end_time - start_time
            
            u_pred = uv_pred[:,0].cpu().numpy()
            v_pred = uv_pred[:,1].cpu().numpy()

    # compute metrics
    metrics_u = Metrics(u_pred,u_gt)
    metrics_v = Metrics(v_pred,v_gt)
    relative_l2_error_value = (metrics_u.relative_l2_error()+metrics_v.relative_l2_error())/2
    relative_l1_error_value = (metrics_u.relative_l1_error()+metrics_v.relative_l1_error())/2

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

    v_pred = v_pred.reshape(n_t_gt,n_x_gt,n_y_gt)
    v_gt = v_gt.reshape(n_t_gt,n_x_gt,n_y_gt)

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    save_path = os.path.join(figs_dir,"plots_u.png")
    plot_prediction_vs_groundtruth(ts=t_gt,u_pred=u_pred,u_gt=u_gt,extent=[*domain_x,*domain_y],save_path=save_path)

    save_path = os.path.join(figs_dir,"plots_v.png")
    plot_prediction_vs_groundtruth(ts=t_gt,u_pred=v_pred,u_gt=v_gt,extent=[*domain_x,*domain_y],save_path=save_path)