import numpy as np
import torch

from utilities.pinnsformer_utils import make_time_sequence, make_sequence_in_first_coord

class ToDeviceMixin:
    def to(self, device):
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device))
            elif isinstance(attr_value, (list, tuple)):
                if all(isinstance(elem, torch.Tensor) for elem in attr_value):
                    converted = type(attr_value)(elem.to(device) for elem in attr_value)
                    setattr(self, attr_name, converted)
        return self

class Regular2DGrid(ToDeviceMixin):
    # creates a regular grid given spacial and time domains and number of points for a 
    # initial-boundary value problem
    def __init__(self,domain_x,domain_t,n_x,n_t):

        self.x_lb, self.x_ub = domain_x
        self.t_lb, self.t_ub = domain_t

        # x as an [n_x,1] tensor
        self.x = torch.linspace(self.x_lb, self.x_ub, n_x).reshape(-1,1)
        # t as an [n_t,1] tensor
        self.t = torch.linspace(self.t_lb, self.t_ub, n_t).reshape(-1,1)

        # [n_x,1] tensor of t_min's to enforce initial condition
        self.t0 = torch.tensor(np.zeros_like(self.x),dtype=torch.float32)

        # [n_t,1] tensor of x_min's to enforce boundary condition
        self.x_lb = torch.tensor(float(self.x_lb)*np.ones_like(self.t),dtype=torch.float32)

        # [n_t,1] tensor of x_max's to enforce boundary condition
        self.x_ub = torch.tensor(float(self.x_ub)*np.ones_like(self.t),dtype=torch.float32)

        # grid points
        X,T = torch.meshgrid(self.x.squeeze(),self.t.squeeze(),indexing='xy')
        X_flat = X.flatten().unsqueeze(1)
        T_flat = T.flatten().unsqueeze(1)
        self.grid_as_column = torch.cat((X_flat,T_flat),dim=1)

        # first (x) and second (t) columns of the complete grid as column
        self.grid_column_x = self.grid_as_column[:,0:1]
        self.grid_column_t = self.grid_as_column[:,1:2]



class Grid2DToTensors():
    # gets x and t values and creates a mesh grid as columns as a torch tensor
    # x and t are expected to have [N,1] shape
    def __init__(self,x,t):

        self.x_tensor = torch.tensor(x,dtype=torch.float32).flatten()
        self.t_tensor = torch.tensor(t,dtype=torch.float32).flatten()


        X,T = torch.meshgrid(self.x_tensor,self.t_tensor, indexing='xy')
        X_flat = X.flatten().unsqueeze(1)
        T_flat = T.flatten().unsqueeze(1)
        
        # complete grid as column: [N,2] tensor, N = n_x * n_t
        self.grid_as_column = torch.cat((X_flat,T_flat),dim=1)
        self.grid_as_column_x = self.grid_as_column[:,0:1]
        self.grid_as_column_t = self.grid_as_column[:,1:2]

class Grid2DPINNsFormer(ToDeviceMixin):
    def __init__(self,domain_x,domain_t,n_x,n_t,num_step=5,step=1e-04):
        
        self.x_min, self.x_max = domain_x
        self.t_min, self.t_max = domain_t

        # x as an [n_x,1] tensor
        self.x = torch.linspace(self.x_min, self.x_max, n_x).reshape(-1,1)
        # t as an [n_t,1] tensor
        self.t = torch.linspace(self.t_min, self.t_max, n_t).reshape(-1,1)

        X,T = torch.meshgrid(self.x.squeeze(),self.t.squeeze(), indexing='xy')
        X_flat = X.flatten().unsqueeze(1)
        T_flat = T.flatten().unsqueeze(1)
        
        # complete grid as column: [N,2] tensor, N = n_x * n_t
        self.grid_as_column = torch.cat((X_flat,T_flat),dim=1)

        t0 = torch.tensor(self.t_min*np.ones_like(self.x),dtype=torch.float32)
        self.initial_condition_points = torch.cat((self.x,t0), dim=1)      # points to enforce initial condition: [n_x,2] tensor
        
        x_min_tensor = torch.tensor(self.x_min*np.ones_like(self.t),dtype=torch.float32)
        self.left_boundary_points = torch.cat((x_min_tensor,self.t),dim=1) # points to enforce left boundary condition: [n_t,2] tensor

        x_max_tensor = torch.tensor(self.x_max*np.ones_like(self.t), dtype=torch.float32)
        self.right_boundary_points = torch.cat((x_max_tensor,self.t), dim=1)  # points to enforce right boundary condition: [n_t,2] tensor

        # NOW WE AUGMENT EVERYTHING with the method make_time_sequence

        self.aug_grid_as_column = make_time_sequence(self.grid_as_column,num_step,step) # [n_x*n_t,num_steps,2]
        self.aug_initial_condition_points = make_time_sequence(self.initial_condition_points,num_step,step) # [n_x, num_steps, 2]
        self.aug_left_boundary_points = make_time_sequence(self.left_boundary_points,num_step,step)   # [n_t, num_steps, 2]
        self.aug_right_boundary_points = make_time_sequence(self.right_boundary_points,num_step,step) # [n_t, num_steps, 2]

        # we separate the coordinates x and t of each augmented tensor above

        self.grid_column_x = self.aug_grid_as_column[:,:,0:1]
        self.grid_column_t = self.aug_grid_as_column[:,:,1:2]

        self.x = self.aug_initial_condition_points[:,:,0:1]
        self.t0 = self.aug_initial_condition_points[:,:,1:2]

        self.x_lb = self.aug_left_boundary_points[:,:,0:1]
        self.t = self.aug_left_boundary_points[:,:,1:2]

        self.x_ub = self.aug_right_boundary_points[:,:,0:1]

class Grid2DPINNsFormerToTensors():
    def __init__(self,x,t):

        self.x = torch.tensor(x,dtype=torch.float32)
        self.t = torch.tensor(t,dtype=torch.float32)

        X,T = torch.meshgrid(self.x.squeeze(),self.t.squeeze(), indexing='xy')
        X_flat = X.flatten().unsqueeze(1)
        T_flat = T.flatten().unsqueeze(1)
        
        # complete grid as column: [N,2] tensor, N = n_x * n_t
        self.grid_as_column = torch.cat((X_flat,T_flat),dim=1)
        self.grid_as_column_x = self.grid_as_column[:,0]
        self.grid_as_column_t = self.grid_as_column[:,1]

        # augmented complete grid: [n_x*n_t,num_steps,2]
        self.aug_grid_as_column = make_time_sequence(self.grid_as_column) 
        self.aug_grid_as_column_x = self.aug_grid_as_column[:,:,0:1]
        self.aug_grid_as_column_t = self.aug_grid_as_column[:,:,1:2]


class Random3DGrid(ToDeviceMixin):
    def __init__(self, domain_x, domain_y, domain_t, n_ic, n_bc, n_res):
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.domain_t = domain_t
        self.n_ic = n_ic
        self.n_bc = n_bc
        self.n_res = n_res

        self.generator = torch.Generator()#.manual_seed(42)

    def update_random_grid(self):

        # generate tensors for initial condition
        x_ic = (self.domain_x[1]-self.domain_x[0]) * torch.rand(self.n_ic, generator=self.generator) + self.domain_x[0]
        y_ic = (self.domain_y[1]-self.domain_y[0]) * torch.rand(self.n_ic, generator=self.generator) + self.domain_y[0]
        t_ic = self.domain_t[0] * torch.ones_like(x_ic)
        self.x_ic = x_ic.reshape(-1,1)
        self.y_ic = y_ic.reshape(-1,1)
        self.t_ic = t_ic.reshape(-1,1)


        # generate tensors for boundary conditions
        t_bc = (self.domain_t[1]-self.domain_t[0]) * torch.rand(self.n_bc, generator=self.generator) + self.domain_t[0]
        x_lb_bc = self.domain_x[0]*torch.ones_like(t_bc)
        x_ub_bc = self.domain_x[1]*torch.ones_like(t_bc)
        x_bc = (self.domain_x[1]-self.domain_x[0]) * torch.rand(self.n_bc, generator=self.generator) + self.domain_x[0]
        y_lb_bc = self.domain_y[0]*torch.ones_like(t_bc)
        y_ub_bc = self.domain_y[1]*torch.ones_like(t_bc)
        y_bc = (self.domain_y[1]-self.domain_y[0]) * torch.rand(self.n_bc, generator=self.generator) + self.domain_y[0]
        self.t_bc = t_bc.reshape(-1,1)
        self.x_lb_bc = x_lb_bc.reshape(-1,1)
        self.x_ub_bc = x_ub_bc.reshape(-1,1)
        self.x_bc = x_bc.reshape(-1,1)
        self.y_lb_bc = y_lb_bc.reshape(-1,1)
        self.y_ub_bc = y_ub_bc.reshape(-1,1)
        self.y_bc = y_bc.reshape(-1,1)

        # generate column grid (x,y,t) for residual loss
        self.x_res = ((self.domain_x[1]-self.domain_x[0]) * torch.rand(self.n_res, generator=self.generator) + self.domain_x[0]).reshape(-1,1)
        self.y_res = ((self.domain_y[1]-self.domain_y[0]) * torch.rand(self.n_res, generator=self.generator) + self.domain_y[0]).reshape(-1,1)
        self.t_res = ((self.domain_t[1]-self.domain_t[0]) * torch.rand(self.n_res, generator=self.generator) + self.domain_t[0]).reshape(-1,1)

class Random3DGridPINNsFormer(ToDeviceMixin):
    def __init__(self, domain_x, domain_y, domain_t, n_ic, n_bc, n_res):
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.domain_t = domain_t
        self.n_ic = n_ic
        self.n_bc = n_bc
        self.n_res = n_res
        self.generator = torch.Generator()
        

    def make_boundary_grid(self,x, y, t):
        grid = make_time_sequence(torch.cat([x, y, t], dim=1))
        return grid[:,:,0:1], grid[:,:,1:2], grid[:,:,2:3] 

    def update_random_grid(self):
        x_ic = ((self.domain_x[1]-self.domain_x[0]) * torch.rand(self.n_ic, generator=self.generator) + self.domain_x[0]).reshape(-1,1)
        y_ic = ((self.domain_y[1]-self.domain_y[0]) * torch.rand(self.n_ic, generator=self.generator) + self.domain_y[0]).reshape(-1,1)
        t_ic = (self.domain_t[0] * torch.ones_like(x_ic)).reshape(-1,1)
        
        self.x_ic, self.y_ic, self.t_ic = self.make_boundary_grid(x_ic,y_ic,t_ic)
        
        t_bc = ((self.domain_t[1]-self.domain_t[0]) * torch.rand(self.n_bc, generator=self.generator) + self.domain_t[0]).reshape(-1,1)
        x_bc = ((self.domain_x[1]-self.domain_x[0]) * torch.rand(self.n_bc, generator=self.generator) + self.domain_x[0]).reshape(-1,1)
        y_bc = ((self.domain_y[1]-self.domain_y[0]) * torch.rand(self.n_bc, generator=self.generator) + self.domain_y[0]).reshape(-1,1)

        x_lb_bc = (self.domain_x[0]*torch.ones_like(t_bc)).reshape(-1,1)
        x_ub_bc = (self.domain_x[1]*torch.ones_like(t_bc)).reshape(-1,1)

        y_lb_bc = (self.domain_y[0]*torch.ones_like(t_bc)).reshape(-1,1)
        y_ub_bc = (self.domain_y[1]*torch.ones_like(t_bc)).reshape(-1,1)

        self.x_lb_bc, self.y_bc, self.t_bc = self.make_boundary_grid(x_lb_bc, y_bc, t_bc)
        self.x_ub_bc, _, _ = self.make_boundary_grid(x_ub_bc, y_bc, t_bc)
        self.x_bc, self.y_lb_bc, _ = self.make_boundary_grid(x_bc, y_lb_bc, t_bc)
        _, self.y_ub_bc, _ = self.make_boundary_grid(x_bc, y_ub_bc, t_bc)

        # generate column grid (x,y,t) for residual loss
        x_res = ((self.domain_x[1]-self.domain_x[0]) * torch.rand(self.n_res) + self.domain_x[0]).reshape(-1,1)
        y_res = ((self.domain_y[1]-self.domain_y[0]) * torch.rand(self.n_res) + self.domain_y[0]).reshape(-1,1)
        t_res = ((self.domain_t[1]-self.domain_t[0]) * torch.rand(self.n_res) + self.domain_t[0]).reshape(-1,1)

        self.x_res, self.y_res, self.t_res = self.make_boundary_grid(x_res,y_res,t_res)
        
class Random4DGrid(ToDeviceMixin):
    def __init__(self,domain_x,domain_y,domain_z,domain_t,n_ic,n_res,n_bc):
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.domain_z = domain_z
        self.domain_t = domain_t
        self.n_ic = n_ic
        self.n_res = n_res
        self.n_bc = n_bc

        self.generator = torch.Generator()

    def update_random_grid(self):
        # generate tensors for initial condition
        x_ic = (self.domain_x[1]-self.domain_x[0]) * torch.rand(self.n_ic, generator=self.generator) + self.domain_x[0]
        y_ic = (self.domain_y[1]-self.domain_y[0]) * torch.rand(self.n_ic, generator=self.generator) + self.domain_y[0]
        z_ic = (self.domain_z[1]-self.domain_z[0]) * torch.rand(self.n_ic, generator=self.generator) + self.domain_z[0]
        t_ic = self.domain_t[0] * torch.ones_like(x_ic)

        self.x_ic = x_ic.reshape(-1,1)
        self.y_ic = y_ic.reshape(-1,1)
        self.z_ic = z_ic.reshape(-1,1)
        self.t_ic = t_ic.reshape(-1,1)

        # generate column grid (x,y,t) for residual loss
        self.x_res = ((self.domain_x[1]-self.domain_x[0]) * torch.rand(self.n_res, generator=self.generator) + self.domain_x[0]).reshape(-1,1)
        self.y_res = ((self.domain_y[1]-self.domain_y[0]) * torch.rand(self.n_res, generator=self.generator) + self.domain_y[0]).reshape(-1,1)
        self.z_res = ((self.domain_z[1]-self.domain_z[0]) * torch.rand(self.n_res, generator=self.generator) + self.domain_z[0]).reshape(-1,1)
        self.t_res = ((self.domain_t[1]-self.domain_t[0]) * torch.rand(self.n_res, generator=self.generator) + self.domain_t[0]).reshape(-1,1)

        # x lower boundary
        _, self.y_bc_x0, self.z_bc_x0, self.t_bc_x0 = self.sample_uniform_points(self.n_bc)
        self.x_bc_x0 = torch.ones(self.n_bc,1) * self.domain_x[0]
        # x upper boundary
        _, self.y_bc_x1, self.z_bc_x1, self.t_bc_x1 = self.sample_uniform_points(self.n_bc)
        self.x_bc_x1 = torch.ones(self.n_bc,1) * self.domain_x[1]
        # y lower boundary
        self.x_bc_y0, _, self.z_bc_y0, self.t_bc_y0 = self.sample_uniform_points(self.n_bc)
        self.y_bc_y0 = torch.ones(self.n_bc,1) * self.domain_y[0]
        # y upper boundary
        self.x_bc_y1, _, self.z_bc_y1, self.t_bc_y1 = self.sample_uniform_points(self.n_bc)
        self.y_bc_y1 = torch.ones(self.n_bc,1) * self.domain_y[1]
        # z lower boundary
        self.x_bc_z0, self.y_bc_z0, _, self.t_bc_z0 = self.sample_uniform_points(self.n_bc)
        self.z_bc_z0 = torch.ones(self.n_bc,1) * self.domain_z[0]
        # z upper boundary
        self.x_bc_z1, self.y_bc_z1, _, self.t_bc_z1 = self.sample_uniform_points(self.n_bc)
        self.z_bc_z1 = torch.ones(self.n_bc,1) * self.domain_z[1]

        
    def sample_uniform_points(self, n_points):
        '''sample n_points points uniformly in a 4D space'''
        x = ((self.domain_x[1]-self.domain_x[0]) * torch.rand(n_points, generator=self.generator) + self.domain_x[0]).reshape(-1,1)
        y = ((self.domain_y[1]-self.domain_y[0]) * torch.rand(n_points, generator=self.generator) + self.domain_y[0]).reshape(-1,1)
        z = ((self.domain_z[1]-self.domain_z[0]) * torch.rand(n_points, generator=self.generator) + self.domain_z[0]).reshape(-1,1)
        t = ((self.domain_t[1]-self.domain_t[0]) * torch.rand(n_points, generator=self.generator) + self.domain_t[0]).reshape(-1,1)
        return x, y, z, t

class Grid3DToTensors:
    def __init__(self,x,y,t):

        x = torch.tensor(x, dtype=torch.float32).flatten()
        y = torch.tensor(y,dtype=torch.float32).flatten()
        t = torch.tensor(t,dtype=torch.float32).flatten()

        T,X,Y = torch.meshgrid(t,x,y, indexing='ij')

        X_flat = X.flatten()  
        Y_flat = Y.flatten()
        T_flat = T.flatten()


        ground_truth_grid = torch.stack([T_flat, X_flat, Y_flat], dim=1)

        self.t = ground_truth_grid[:,0:1]
        self.x = ground_truth_grid[:,1:2]
        self.y = ground_truth_grid[:,2:3]

class Grid3DPINNsFormerToTensors():
    def __init__(self,x,y,t):

        x = torch.tensor(x,dtype=torch.float32).flatten()
        t = torch.tensor(t,dtype=torch.float32).flatten()
        y = torch.tensor(y,dtype=torch.float32).flatten()

        T,X,Y = torch.meshgrid(t,x,y, indexing='ij')
        T_flat = T.flatten().unsqueeze(1)
        X_flat = X.flatten().unsqueeze(1)
        Y_flat = Y.flatten().unsqueeze(1)
        
        
        # complete grid as column: [N,2] tensor, N = n_x * n_t
        self.grid_as_column = torch.cat([T_flat,X_flat,Y_flat],dim=1)
        self.grid_as_column_t = self.grid_as_column[:,0:1]
        self.grid_as_column_x = self.grid_as_column[:,1:2]
        self.grid_as_column_y = self.grid_as_column[:,2:3]

        # augmented complete grid: [n_x*n_t,num_steps,2]
        self.aug_grid_as_column = make_sequence_in_first_coord(self.grid_as_column) 
        self.aug_grid_as_column_t = self.aug_grid_as_column[:,:,0:1]
        self.aug_grid_as_column_x = self.aug_grid_as_column[:,:,1:2]
        self.aug_grid_as_column_y = self.aug_grid_as_column[:,:,2:3]
