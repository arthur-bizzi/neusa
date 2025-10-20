import torch
from torchdyn.core import NeuralODE
import time
import json
import argparse
import os

from basis.cosine2d import CosineBasis2D
from models.hadamard_layer import HadamardLayer
from models.vector_fields import WaveLikeVectorField
from models.mlps import MLPWithTranspose2D
from utilities.trainer import AdamTrainer
from data_loaders.wave2d_three_layers_data_loader import Wave2DThreeLayersGroundTruthLoader
from utilities.metrics import Metrics
from utilities.plot import plot_prediction_vs_groundtruth, plot_error

def loss_function(NODE,basis,huv0,ts,F,F0,c2):
    huv = NODE(huv0, ts)
    Fhuv = F(huv)

    hu, _ = huv.chunk(2, dim=-1)
    _, Fhv = Fhuv.chunk(2, dim=-1)

    right_hand_side = c2 * basis.idbt(F0(hu))
    v_t = basis.idbt(Fhv)

    loss = torch.mean((v_t-right_hand_side)**2)

    return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Wave 2D three layers NeuSA')
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

    save_dir = os.path.join("results", "wave2d_three_layers", dir_name, "NeuSA", f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    # baseline domain
    domain_x = [-2.0,2.0]
    domain_y = [-2.0,2.0]
    domain_t = [0.0,2.0]
    n_x = 101
    n_y = 101
    n_t = 201
    
    ts = torch.linspace(*domain_t, n_t)

    # extended domain for training
    domain_x_ext = [-4.0,4.0]
    domain_y_ext = [-4.0,4.0]
    M = 201
    N = 201
    
    # basis and 2 dimensional spatial grid
    basis = CosineBasis2D(M,N,domain_x_ext,domain_y_ext,device=device)
    X,Y = torch.meshgrid(basis.x,basis.y,indexing='ij')
    
    grid = torch.stack((X,Y),dim=-1)    # shape = (M,N,2)
    xs = grid[:,:,0:1]
    ys = grid[:,:,1:2]
    
    # initial condition
    sigma = 0.1
    u0 = torch.exp(-((xs)**2+(ys)**2)/(2*sigma**2)).squeeze(-1).to(device)
    hu0 = basis.dbt(u0)     # shape = (M,N)
    hv0 = torch.zeros_like(hu0)
    huv0 = torch.cat([hu0,hv0],dim=-1) # shape = (M,2N)
    
    # velocity field
    c = (1.0 + 0.25 * (torch.sigmoid(1000 * (ys - 0.5)) + torch.sigmoid(1000 * (ys - 1.0)))).squeeze(-1)
    c2 = c*c
    
    # initial vector field (Laplacian in the frequency domain as Hadamard product)
    omegas = basis.D2
    F0 = HadamardLayer(omegas)

    # neural network and NODE
    F_neural = MLPWithTranspose2D(M=M,N=N)
    F = WaveLikeVectorField(F_neural,F0,model_weight=1.0)
    NODE = NeuralODE(F,solver='rk4',sensitivity='autograd',return_t_eval=False)

    # training
    F = F.to(device)
    NODE = NODE.to(device)

    basis = basis.to(device)
    huv0 = huv0.to(device)
    ts = ts.to(device)
    F0 = F0.to(device)
    c2 = c2.to(device)

    trainer = AdamTrainer(NODE.parameters(),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function= lambda: loss_function(NODE,basis,huv0,ts,F,F0,c2)
                )
    
    start_time = time.time()
    trainer.train_neusa()
    end_time = time.time()

    training_time = end_time - start_time

    # save trained model
    torch.save(F.state_dict(),os.path.join(save_dir, "model.pt"))

    # retrieve model solution
    start_time = time.time()
    huv = NODE(huv0, ts)
    end_time = time.time()
    inference_time = end_time-start_time

    hu, _ = huv.chunk(2, dim=-1)

    u_pred = basis.idbt(hu).squeeze().detach()
    u_pred = u_pred[:, 50:151, 50:151].cpu().numpy()

    # load ground truth solution
    data_loader = Wave2DThreeLayersGroundTruthLoader()
    x_gt,y_gt,t_gt = data_loader.get_ground_truth_grid()
    u_gt = data_loader.get_ground_truth_solution()
    n_x_gt, n_y_gt, n_t_gt = data_loader.get_grid_size()
    x,y,t = data_loader.get_ground_truth_grid_as_tensors()

    u_gt = u_gt.reshape(n_t_gt,n_x_gt,n_y_gt)

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
    
     ### PLOTS ###

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    save_path = os.path.join(figs_dir,"plots.png")
    plot_prediction_vs_groundtruth(ts=ts,u_pred=u_pred,u_gt=u_gt,extent=[*domain_x,*domain_y],save_path=save_path,cmap="seismic")
    
    save_path = os.path.join(figs_dir,"error.png")
    plot_error(ts=ts,error=u_pred-u_gt,extent=[*domain_x,*domain_y],save_path=save_path,cmap="seismic")











