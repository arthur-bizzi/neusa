import torch

class FourierBasis1D:
    def __init__(self,M,domain,device="cpu"):

        self.M = M      # number of points in the discretization of the physical domain
        self.N = int((M+1)/2)       
        self.device=device

        # cosine frequencies are 0 to N-1, and sine frequencies are 1 to N-1
        self.idx_freqs_cos = torch.arange(0,self.N)
        self.idx_freqs_sin = torch.arange(1,self.N)
        self.idx_freqs = torch.arange(0,self.M)

        self.z = torch.linspace(0,2*torch.pi,2*self.N,device=device)[:-1]
        # (0,2pi/M,4pi/M,...,2pi(M-1)/M)

        self.a, self.b = domain

        self.x = (self.b - self.a) * self.z / (2*torch.pi) + self.a
        self.jacobian = 2 * torch.pi / (self.b - self.a)


        self.Cx = torch.zeros((self.M,self.M),device=self.device)
        for i in range(self.N):
            for j in range(self.M):
                self.Cx[i,j] = torch.cos(i * self.z[j])
        
        for i in range(self.N,self.M):
            for j in range(self.M):
                self.Cx[i,j] = torch.sin((i-self.N+1) * self.z[j])

        
        self.Cx_inv = torch.inverse(self.Cx)

        # To derivate, multiply D ON THE RIGHT of u_hat.
        self.D = self.jacobian * self.differentiation_matrix()
        
        # Second derivative can be obtained by multiplying D2 below
        # ON THE RIGHT of u_hat, but in practice we use a Hadamard product
        # with the diagonal elements of D2
        self.D2 = self.D @ self.D

        self.to(device)


    def dbt(self,u):
        u_hat = u @ self.Cx_inv
        return u_hat

    def idbt(self,u_hat):
        u = u_hat @ self.Cx
        return u

    def differentiation_matrix(self):

        zero_matrix = torch.zeros((self.N-1,self.N-1))
        A_matrix = torch.zeros((self.N-1, self.N-1))

        for i in range(self.N-1):
            A_matrix[i,i] = -i-1
        
        upper_block = torch.cat((zero_matrix,A_matrix),dim=1)
        lower_block = torch.cat((-A_matrix,zero_matrix),dim=1)

        matrix = torch.cat((upper_block,lower_block),dim=0)

        upper_line = torch.zeros(1,2*self.N-2)
        left_column = torch.zeros(2*self.N-1,1)

        matrix = torch.cat((upper_line,matrix),dim=0)
        matrix = torch.cat((left_column,matrix),dim=1)

        return matrix   


    def to(self,device):
        self.Cx = self.Cx.to(device)
        self.Cx_inv = self.Cx_inv.to(device)
        self.D = self.D.to(device)
        self.D2 = self.D2.to(device)
        self.device = device
        
        return self



