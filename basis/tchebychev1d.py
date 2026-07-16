import torch

class TchebychevBasis1D:
    def __init__(self, M, domain, device="cpu"):

        self.M = M
        self.device = device

        self.domain = domain

        k = torch.arange(M, device=device)
        self.z = torch.cos((2*k+1)/(2*M)*torch.pi)

        self.a, self.b = domain
        self.x = ((self.b-self.a)/2)*self.z + ((self.a+self.b)/2)

        self.jacobian = 2 / (self.b-self.a)

        self.Vx = torch.zeros((self.M,self.M), device=self.device)
        for i in range(self.M):
            self.Vx[i,0] = 1
        
        for i in range(self.M):
            self.Vx[i,1] = self.z[i]
        
        for i in range(self.M):
            for j in range(2,self.M):
                self.Vx[i,j] = 2*self.z[i]*self.Vx[i,j-1]-self.Vx[i,j-2]
        

        d_inv = torch.full((M,), 2.0/M, device=device)
        d_inv[0] = 1.0/M

        self.Vx_inv = self.Vx * d_inv
        self.Vx = self.Vx.T.contiguous ()

        self.D = self.jacobian * self.differentiation_matrix()

        self.D2 = self.D @ self.D 

        self.to(device)

    def dbt(self,u):
        u_hat = u @ self.Vx_inv
        return u_hat
    
    def idbt(self,u_hat):
        u = u_hat @ self.Vx
        return u

    def differentiation_matrix(self):

        matrix  = torch.zeros((self.M,self.M))

        for i in range(self.M):
            matrix[self.M-1,i]=0


        matrix[self.M-2,self.M-1]=2*self.M

        for i in range(self.M-3, -1, -1):
            for j in range(self.M):
                matrix[i,j]=matrix[i+2,j]
            matrix[i,i+1]+=2*(i+1)
        
        return matrix

    def to(self, device):
        self.Vx = self.Vx.to(device)
        self.Vx_inv = self.Vx_inv.to(device)
        self.D = self.D.to(device)
        self.D2 = self.D2.to(device)
        self.device = device

        return self

class TchebychevBasisTruncated1D:
    def __init__(self, M, K, domain, device="cpu"):

        self.M = M 
        self.K= K
        self.P = K + 1

        self.a, self.b = domain

        k = torch.arange(M, device=device)
        self.z = torch.cos((2*k+1)/(2*M)*torch.pi)

        self.a, self.b = domain
        self.x = ((self.b-self.a)/2)*self.z + ((self.a+self.b)/2)

        self.jacobian = 2 / (self.b-self.a)

        V = torch.zeros((M, self.P), device=device)
        V[:, 0] = 1.0
        if self.P > 1:
            V[:, 1] = self.z
        
        for j in range(2, self.P):
            V[:, j] = 2*self.z*V[:, j-1] - V[:,j-2]
        
        d_inv = torch.full((self.P,), 2.0/M, device=device)
        d_inv[0] = 1.0/M 
        self.Vx_inv = V * d_inv
        self.Vx = V.T.contiguous()

        self.D = self.jacobian * self.differentiation_matrix(self.P)
        self.D2 = self.D @ self.D

        self.to(device)

    def dbt(self,u):
        u_hat = u @ self.Vx_inv
        return u_hat
    
    def idbt(self, u_hat):
        u = u_hat @ self.Vx
        return u

    def differentiation_matrix(self,N):
        matrix  = torch.zeros((N,N))

        for i in range(N):
            matrix[N-1,i]=0


        matrix[N-2,N-1]=2*N

        for i in range(N-3, -1, -1):
            for j in range(N):
                matrix[i,j]=matrix[i+2,j]
            matrix[i,i+1]+=2*(i+1)
        
        return matrix
    
    def to(self, device):
        self.Vx = self.Vx.to(device)
        self.Vx_inv = self.Vx_inv.to(device)
        self.D = self.D.to(device)
        self.D2 = self.D2.to(device)
        self.device = device

        return self