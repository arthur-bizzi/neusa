import torch

class SineBasis1D:
    def __init__(self,M,domain,device="cpu"):
        self.device = device
        
        self.M = M
        self.a, self.b = domain

        self.idx_freqs = torch.arange(1,self.M+1)

        self.z = torch.pi / (self.M+1) *  self.idx_freqs
        self.x = self.z * ((self.b-self.a)/torch.pi) + self.a


        self.Cx = torch.zeros((self.M,self.M), device=self.device)
        for i in range(self.M):
            for j in range(self.M):
                self.Cx[i,j] = torch.sin((i+1)*self.z[j])

        self.Cx_inv = torch.inverse(self.Cx)

        self.D2 = torch.zeros((self.M,self.M),device=self.device)
        for i in range(self.M):
            self.D2[i,i] = -(torch.pi / (self.b-self.a))**2 * (i+1)**2

        self.to(device)

    
    def dbt(self,u):
        u_hat = u @ self.Cx_inv
        return u_hat

    def idbt(self,u_hat):
        u = u_hat @ self.Cx
        return u

    def to(self,device):
        self.Cx = self.Cx.to(device)
        self.Cx_inv = self.Cx_inv.to(device)
        self.D2 = self.D2.to(device)
        self.device = device

        return self