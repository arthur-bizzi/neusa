import os
import numpy as np
from utilities.grid import Grid2DToTensors, Grid2DPINNsFormerToTensors

class KleinGordonTriangular1DGroundTruthLoader:
    def __init__(self, model_name="NeuSA"):

        self.model_name = model_name
        self.data_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "klein_gordon_triangular")
        self.x_file_path = os.path.join(self.data_dir_path,"x.txt")
        self.x = np.loadtxt(self.x_file_path)

        self.t_file_path = os.path.join(self.data_dir_path,"t.txt")
        self.t = np.loadtxt(self.t_file_path)

    def get_ground_truth_grid(self):
        return self.x,self.t
    
    def get_ground_truth_solution(self):
        self.u_file_path = os.path.join(self.data_dir_path,"u.txt")
        self.u = np.loadtxt(self.u_file_path)

        return self.u.flatten()[:,None].ravel()
    
    def get_ground_truth_grid_as_tensors(self):

        if self.model_name == "PINNsFormer":
            grid_ground_truth = Grid2DPINNsFormerToTensors(self.x,self.t)
            x_tensor = grid_ground_truth.aug_grid_as_column_x
            t_tensor = grid_ground_truth.aug_grid_as_column_t
        else:
            grid_ground_truth = Grid2DToTensors(self.x,self.t)
            x_tensor = grid_ground_truth.grid_as_column_x
            t_tensor = grid_ground_truth.grid_as_column_t

        return x_tensor, t_tensor
    
    def get_grid_size(self):
        return len(self.t), len(self.x)