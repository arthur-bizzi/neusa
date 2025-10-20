import torch

class FourierBasis2D:
    def __init__(self, Mx, My, domain_x, domain_y, device="cpu"):
        self.device = device
        
        self.Mx = Mx
        self.My = My

        self.Nx = int((Mx+1)/2)
        self.Ny = int((My+1)/2)

        self.a, self.b = domain_x
        self.c, self.d = domain_y

        self.idx_freqs_x = torch.arange(0,self.Mx)
        self.idx_freqs_y = torch.arange(0,self.My)

        self.z = ((2*torch.pi) / self.Mx) * self.idx_freqs_x
        self.w = ((2*torch.pi) / self.My) * self.idx_freqs_y

        self.x = ((self.b - self.a) / (2*torch.pi)) * self.z + self.a 
        self.y = ((self.d - self.c) / (2*torch.pi)) * self.w + self.c

        self.jacobian_x = 2*torch.pi / (self.b - self.a)
        self.jacobian_y = 2*torch.pi / (self.d - self.c)

        self.Cx = torch.zeros((self.Mx,self.Mx))
        for i in range(self.Mx):
            for j in range(self.Nx):
                self.Cx[i,j] = torch.cos(j * self.z[i])
        
        for i in range(self.Mx):
            for j in range(self.Nx,self.Mx):
                self.Cx[i,j] = torch.sin((j-self.Nx+1) * self.z[i])
        
        self.Cy = torch.zeros((self.My,self.My))
        for i in range(self.My):
            for j in range(self.Ny):
                self.Cy[i,j] = torch.cos(j * self.w[i])
        
        for i in range(self.My):
            for j in range(self.Ny,self.My):
                self.Cy[i,j] = torch.sin((j-self.Ny+1) * self.w[i])

        self.Cx_inv = torch.inverse(self.Cx)
        self.Cy_inv = torch.inverse(self.Cy)

        # To derivate with respect to x, multiply Dx (D2x) ON THE LEFT of u_hat
        self.Dx = self.jacobian_x * self.differentiation_matrix(self.Nx)
        self.D2x = self.Dx @ self.Dx

        # To derivate with respect to y, multiply Dy (D2y) ON THE RIGHT of u_hat
        self.Dy = self.jacobian_y * self.differentiation_matrix(self.Ny).T
        self.D2y = self.Dy @ self.Dy

        self.omegas_x = torch.diag(self.D2x)
        self.omegas_y = torch.diag(self.D2y)

        # The Laplacian is the HADAMARD PRODUCT of D2 and u_hat
        self.D2 = torch.zeros((self.Mx,self.My))
        for k in range(self.Mx):
            for l in range(self.My):
                self.D2[k,l] = self.omegas_x[k] + self.omegas_y[l]
        
        self.to(device)


    def dbt(self,u):
        u_hat = torch.einsum("in,...nm,jm->...ij", self.Cx_inv,u,self.Cy_inv)
        return u_hat

    def idbt(self,u_hat):
        u = torch.einsum("in,...nm,jm->...ij", self.Cx,u_hat,self.Cy)
        return u
    
    def differentiation_matrix(self,N):
        zero_matrix = torch.zeros((N-1,N-1))
        A_matrix = torch.zeros((N-1,N-1))

        for i in range(N-1):
            A_matrix[i,i] = i+1

        upper_block = torch.cat((zero_matrix,A_matrix),dim=1)
        lower_block = torch.cat((-A_matrix,zero_matrix),dim=1)

        matrix = torch.cat((upper_block,lower_block),dim=0)

        upper_line = torch.zeros(1,2*N-2)
        left_column = torch.zeros(2*N-1,1)

        matrix = torch.cat((upper_line,matrix),dim=0)
        matrix = torch.cat((left_column,matrix),dim=1)

        return matrix

    def to(self,device):
        self.Cx = self.Cx.to(device)
        self.Cx_inv = self.Cx_inv.to(device)
        self.Cy = self.Cy.to(device)
        self.Cy_inv = self.Cy_inv.to(device)
        self.Dx = self.Dx.to(device)
        self.Dy = self.Dy.to(device)
        self.D2x = self.D2x.to(device)
        self.D2y = self.D2y.to(device)
        self.D2 = self.D2.to(device)
        self.device = device

        return self