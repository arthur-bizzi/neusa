import torch

class TchebychevBasis2D:
    def __init__(self, Mx, My, domain_x, domain_y, device="cpu"):
        self.device = device

        self.Mx = Mx
        self.My = My

        self.a, self.b = domain_x
        self.c, self.d = domain_y

        kx = torch.arange(Mx, device=device)
        ky = torch.arange(My, device=device)
        self.z = torch.cos((2*kx+1)/(2*Mx)*torch.pi)
        self.w = torch.cos((2*ky+1)/(2*My)*torch.pi)

        self.x = ((self.b-self.a)/2)*self.z + ((self.a+self.b)/2)
        self.y = ((self.c-self.d)/2)*self.w + ((self.c+self.d)/2)

        self.jacobian_x = 2 / (self.b - self.a)
        self.jacobian_y = 2 / (self.d - self.c)

        self.Vx = torch.zeros((self.Mx,self.Mx), device=self.device)
        for i in range(self.Mx):
            self.Vx[i,0] = 1
        
        for i in range(self.Mx):
            self.Vx[i,1] = self.z[i]
        
        for i in range(self.Mx):
            for j in range(2,self.Mx):
                self.Vx[i,j] = 2*self.z[i]*self.Vx[i,j-1]-self.Vx[i,j-2]
        
        self.Vy = torch.zeros((self.My,self.My), device=self.device)
        for i in range(self.My):
            self.Vy[i,0] = 1
        
        for i in range(self.My):
            self.Vy[i,1] = self.w[i]
        
        for i in range(self.My):
            for j in range(2,self.My):
                self.Vy[i,j] = 2*self.w[i]*self.Vy[i,j-1]-self.Vy[i,j-2]
        
        d_inv_x = torch.full((Mx,), 2.0/Mx, device=device)
        d_inv_x[0] = 1.0/Mx

        self.Vx_inv = self.Vx * d_inv_x
        self.Vx = self.Vx.T.contiguous ()

        d_inv_y = torch.full((My,), 2.0/My, device=device)
        d_inv_y[0] = 1.0/My

        self.Vy_inv = self.Vy * d_inv_y
        self.Vy = self.Vy.T.contiguous ()

        self.Dx = self.jacobian_x * self.differentiation_matrix(self.Mx)
        self.D2x = self.Dx @ self.Dx

        self.Dy = self.jacobian_y * self.differentiation_matrix(self.My)
        self.D2y = self.Dy @ self.Dy

        self.D2 = self.D2x + self.D2y

        self.to(device)

    
    def dbt(self,u):
        u_hat = self.Vx_inv.T @ u @ self.Vy_inv
        return u_hat
    
    def idbt(self, u_hat):
        u = self.Vx.T @ u_hat @ self.Vy
        return u


    def differentiation_matrix(self, N):

        matrix  = torch.zeros(N,N)

        for i in range(N):
            matrix[N-1,i]=0


        matrix[N-2,N-1]=2*N

        for i in range(N-3, -1, -1):
            for j in range(N):
                matrix[i,j]=matrix[i+2,j]
            matrix[i,i+1]+=2*(i+1)
        
        return matrix
    
    def to(self,device):
        self.Vx = self.Vx.to(device)
        self.Vx_inv = self.Vx_inv.to(device)
        self.Vy = self.Vy.to(device)
        self.Vy_inv = self.Vy_inv.to(device)
        self.Dx = self.Dx.to(device)
        self.Dy = self.Dy.to(device)
        self.D2x = self.D2x.to(device)
        self.D2y = self.D2y.to(device)
        self.D2 = self.D2.to(device)
        self.device = device

        return self