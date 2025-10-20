import torch


class SineBasis2D:
    def __init__(self,M,N,domain_x,domain_y,device="cpu"):
        
        self.device = device
        self.M = M
        self.N = N
        self.a, self.b = domain_x
        self.c, self.d = domain_y

        self.z = torch.pi/(self.M + 1) * torch.arange(1,M+1)
        self.w = torch.pi/(self.N + 1) * torch.arange(1,N+1)

        self.x = (self.b - self.a)/torch.pi * self.z + self.a
        self.y = (self.d - self.c)/torch.pi * self.w + self.c

        self.jacobian_x = torch.pi/(self.b - self.a)
        self.jacobian_y = torch.pi/(self.d - self.c)

        self.Cx = torch.zeros((self.M,self.M))
        for i in range(self.M):
            for j in range(self.M):
                self.Cx[i,j] = torch.sin((j+1) * self.z[i]) 

        self.Cy = torch.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.Cy[i,j] = torch.sin((j+1) * self.w[i])
        
        self.Cx_inv = torch.inverse(self.Cx)
        self.Cy_inv = torch.inverse(self.Cy)

        # D2 is a matrix whose Hadamard product with u_hat gives the Laplacian in the frequency domain
        self.D2 = torch.zeros((self.M,self.N))
        for k in range(self.M):
            for l in range(self.N):
                self.D2[k,l] = -((k+1)*self.jacobian_x)**2 - ((l+1)*self.jacobian_y)**2
        
        self.to(device)
        

    def dbt(self,u):
        u_hat = torch.einsum('in,...nm,...jm->ij',self.Cx_inv,u,self.Cy_inv)
        return u_hat

    def idbt(self,u_hat):
        u = torch.einsum('in,...nm,...jm->ij',self.Cx,u_hat,self.Cy)
        return u

    def to(self,device):
        self.Cx = self.Cx.to(device)
        self.Cx_inv = self.Cx_inv.to(device)
        self.Cy = self.Cy.to(device)
        self.Cy_inv = self.Cy_inv.to(device)
        self.D2 = self.D2.to(device)
        self.device = device

        return self
