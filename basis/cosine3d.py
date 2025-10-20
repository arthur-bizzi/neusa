import torch


class CosineBasis3D:
    def __init__(self,Mx,My,Mz,domain_x,domain_y,domain_z,device="cpu"):
        self.device = device
        
        self.Mx = Mx
        self.My = My
        self.Mz = Mz

        self.ax,self.bx = domain_x
        self.ay,self.by = domain_y
        self.az,self.bz = domain_z

        self.jacobian_x = torch.pi/(self.bx - self.ax)
        self.jacobian_y = torch.pi/(self.by - self.ay)
        self.jacobian_z = torch.pi/(self.bz - self.az)

        self.idx_freqs_x = torch.arange(0,self.Mx)
        self.idx_freqs_y = torch.arange(0,self.My)
        self.idx_freqs_z = torch.arange(0,self.Mz)

        self.fx = (torch.pi/(self.Mx-1)) * self.idx_freqs_x
        self.fy = (torch.pi/(self.My-1)) * self.idx_freqs_y
        self.fz = (torch.pi/(self.Mz-1)) * self.idx_freqs_z

        self.x = ((self.bx-self.ax)/torch.pi) * self.fx + self.ax
        self.y = ((self.by-self.ay)/torch.pi) * self.fy + self.ay
        self.z = ((self.bz-self.az)/torch.pi) * self.fz + self.az


        self.Cx = torch.zeros((self.Mx,self.Mx))
        for i in range(self.Mx):
            for j in range(self.Mx):
                self.Cx[i,j] = torch.cos(i * self.fx[j])
        
        self.Cy = torch.zeros((self.My,self.My))
        for i in range(self.My):
            for j in range(self.My):
                self.Cy[i,j] = torch.cos(i * self.fy[j])

        self.Cz = torch.zeros((self.Mz,self.Mz))
        for i in range(self.Mz):
            for j in range(self.Mz):
                self.Cz[i,j] = torch.cos(i * self.fz[j])

        self.Cx_inv = torch.inverse(self.Cx)
        self.Cy_inv = torch.inverse(self.Cy)
        self.Cz_inv = torch.inverse(self.Cz)
        

        # Laplacian multipliers (frequency domain) to be implemented as Hadamard product
        self.D2 = torch.zeros((self.Mx,self.My,self.Mz))
        for k in range(self.Mx):
            for l in range(self.My):
                for m in range(self.Mz):
                    self.D2[k,l,m] = -(k * self.jacobian_x)**2 - (l * self.jacobian_y)**2 - (m * self.jacobian_z)**2

        # Second order derivatives in the frequency domain to be implemented as matrix product
        self.D2x = torch.zeros((self.Mx,self.Mx))
        for i in range(self.Mx):
            self.D2x[i,i] = -(self.jacobian_x * i)**2

        self.D2y = torch.zeros((self.My,self.My))
        for i in range(self.My):
            self.D2y[i,i] = -(self.jacobian_y * i)**2
        
        self.D2z = torch.zeros((self.Mz,self.Mz))
        for i in range(self.Mz):
            self.D2z[i,i] = -(self.jacobian_z * i)**2
        
        self.to(device)


    def dbt(self,u):
        u_hat = torch.einsum("...nml,ni,mj,lk->...ijk", u, self.Cx_inv, self.Cy_inv, self.Cz_inv)
        return u_hat
    
    def idbt(self,u_hat):
        u = torch.einsum("...nml,ni,mj,lk->...ijk", u_hat, self.Cx, self.Cy, self.Cz)
        return u

    def to(self,device):
        self.Cx = self.Cx.to(device)
        self.Cy = self.Cy.to(device)
        self.Cz = self.Cz.to(device)
        self.Cx_inv = self.Cx_inv.to(device)
        self.Cy_inv = self.Cy_inv.to(device)
        self.Cz_inv = self.Cz_inv.to(device)
        self.D2 = self.D2.to(device)
        self.D2x = self.D2x.to(device)
        self.D2y = self.D2y.to(device)
        self.D2z = self.D2z.to(device)
        self.device = device

        return self