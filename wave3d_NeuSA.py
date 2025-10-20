import torch
from torchdyn.core import NeuralODE
import time
import json
import argparse
import os
import numpy as np

from basis.cosine3d import CosineBasis3D
from models.hadamard_layer import HadamardLayer
from models.vector_fields import WaveLikeVectorField
from models.mlps import MLPWithTranspose3D
from utilities.trainer import AdamTrainer
from utilities.metrics import Metrics
from data_loaders.wave3d_data_loader import wave3d_groundtruth_generator
from utilities.plot import generate_figures_3d

def loss_function(NODE,F,F0,c2,huv0,ts,basis):
    huv = NODE(huv0,ts)
    Fhuv = F(huv)
    hu,_ = huv.chunk(2,dim=-1)
    _,Fhv = Fhuv.chunk(2,dim=-1)

    right_side = c2 * basis.idbt(F0(hu))
    v_t = basis.idbt(Fhv)

    loss = torch.mean((v_t-right_side)**2)

    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Wave 3D NeuSA')
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')   
    parser.add_argument('--steps', type=int, default=2000)  
    parser.add_argument('--lr', type=float, default=0.01)
    
    args = parser.parse_args()
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    save_dir = os.path.join("results", "wave3d", dir_name, "NeuSA", f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    # domain
    domain_x = [-1.5,1.5]
    domain_y = [-1.5,1.5]
    domain_z = [-1.5,1.5]
    domain_t = [0.0,1.0]

    # resolution
    n_x = 101
    n_y = 101
    n_z = 101
    n_t = 111

    # basis and grid
    basis = CosineBasis3D(n_x,n_y,n_z,domain_x,domain_y,domain_z,device=device)
    X,Y,Z = torch.meshgrid(basis.x,basis.y,basis.z, indexing='ij')
    grid = torch.stack((X,Y,Z),dim=-1)
    xs = grid[:,:,:,0:1]
    ys = grid[:,:,:,1:2]
    zs = grid[:,:,:,2:3]

    ts = torch.linspace(*domain_t,n_t)

    # initial condition
    sigma = 0.15
    u0 = 10*torch.exp((-xs**2-ys**2-zs**2)/(2*sigma**2)).squeeze(-1).to(device)
    hu0 = basis.dbt(u0)
    hv0 = torch.zeros_like(hu0)
    huv0 = torch.cat([hu0,hv0],dim=-1)

    # velocity field
    c = (1.0 - 0.5 * (torch.sigmoid(1000 * (zs - 0.5)))).squeeze(-1)
    c2 = c*c

    # initial vector field (Laplacian in the frequency domain as Hadamard product)
    omegas = basis.D2
    F0 = HadamardLayer(omegas)

    # neural network and NODE
    F_neural = MLPWithTranspose3D(M=n_x,N=n_y,P=n_z)
    F = WaveLikeVectorField(F_neural,F0,model_weight=10.0)
    NODE = NeuralODE(F,solver='rk4',sensitivity='autograd',return_t_eval=False)

    # training
    huv0 = huv0.to(device)
    c2 = c2.to(device)
    ts = ts.to(device)
    NODE = NODE.to(device)
    basis = basis.to(device)
    F = F.to(device)
    F0 = F0.to(device)

    trainer = AdamTrainer(NODE.parameters(),
                            adam_lr = lr,
                            adam_steps = steps,
                            learning_rate_scheduler=True,
                            loss_function= lambda: loss_function(NODE,F,F0,c2,huv0,ts,basis)
                        )
    
    start_time = time.time()
    trainer.train_neusa()
    end_time = time.time()

    training_time = end_time - start_time

    # save trained model
    torch.save(F.state_dict(), os.path.join(save_dir, "model.pt"))

    # retrieve model solution
    start_time = time.time()
    huv = NODE(huv0,ts)
    end_time = time.time()
    inference_time = end_time-start_time
    hu,_ = huv.chunk(2,dim=-1)
    u_pred = basis.idbt(hu).detach().cpu().numpy()

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

    xs = np.linspace(*domain_x,n_x)
    ys = np.linspace(*domain_y,n_y)
    zs = np.linspace(*domain_z,n_z)
    ts = np.linspace(*domain_t,n_t)

    u_pred = np.expand_dims(u_pred,axis=-1)
    u_gt = np.expand_dims(u_gt,axis=-1)

    generate_figures_3d(ts,xs,ys,zs,u_pred,u_gt,save_dir_figs=figs_dir,order="txyz")


