import os
import numpy as np
import scipy.io
from utilities.grid import Grid2DToTensors, Grid2DPINNsFormerToTensors

class Burgers1DGroundTruthLoader:
    def __init__(self, model_name="NeuSA"):
        self.model_name = model_name
        self.file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data","burgers1d","burgers_shock.mat")
        
        self.data = scipy.io.loadmat(self.file_path)

        self.x = self.data['x'] # (256,)
        self.t = self.data['t'] # (100, )
        self.u = np.real(self.data['usol']).T # (100,256)
    
    def get_ground_truth_grid(self):
        return self.x, self.t
    
    def get_ground_truth_solution(self):
        return self.u.flatten()
    
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
        return len(self.x), len(self.t)