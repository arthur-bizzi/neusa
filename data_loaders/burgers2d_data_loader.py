import os
import numpy as np

from utilities.grid import Grid3DToTensors, Grid3DPINNsFormerToTensors

class Burgers2DGroundTruthLoader:
    def __init__(self, model_name="NeuSA", dirname="burgers2d"):
        self.model_name = model_name
        self.data_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dirname)

        self.x_file_path = os.path.join(self.data_dir_path,"x.txt")
        self.x = np.loadtxt(self.x_file_path)

        self.y_file_path = os.path.join(self.data_dir_path,"y.txt")
        self.y = np.loadtxt(self.y_file_path)
    
        self.t_file_path = os.path.join(self.data_dir_path,"t.txt")
        self.t = np.loadtxt(self.t_file_path)
        
    def get_ground_truth_grid(self):
        return self.x,self.y,self.t

    def get_ground_truth_solution_u(self):
        self.u_file_path = os.path.join(self.data_dir_path,"u.txt")
        self.u = np.loadtxt(self.u_file_path)

        return self.u
    
    def get_ground_truth_solution_v(self):
        self.v_file_path = os.path.join(self.data_dir_path,"v.txt")
        self.v = np.loadtxt(self.v_file_path)

        return self.v
    
    def get_ground_truth_grid_as_tensors(self):

        if self.model_name == "PINNsFormer":
            grid_as_burgers2d_groundtruth_generatortensors = Grid3DPINNsFormerToTensors(self.x,self.y, self.t)  
            x_tensor = grid_as_tensors.aug_grid_as_column_x
            y_tensor = grid_as_tensors.aug_grid_as_column_y
            t_tensor = grid_as_tensors.aug_grid_as_column_t

        else:
            
            grid_as_tensors = Grid3DToTensors(self.x,self.y,self.t)

            x_tensor = grid_as_tensors.x
            y_tensor = grid_as_tensors.y
            t_tensor = grid_as_tensors.t

        return x_tensor, y_tensor, t_tensor
    
    def get_grid_size(self):
        return len(self.x), len(self.y), len(self.t)
