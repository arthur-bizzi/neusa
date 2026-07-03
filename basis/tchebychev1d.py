import torch

class TchebychevBasis1D:
    def __init__(self, M, domain, device="cpu"):

        self.M = M
        self.device = device

        self.domain = domain

        k = torch.arange(M, dtype=torch.float64, device=device)
        self.z = torch.cos((2*k+1)/(2*M)*torch.pi)   # note: /(2M), pas /(2M+2)

        self.a, self.b = domain
        self.z = ((self.b-self.a)/2)*self.z + ((self.a+self.b)/2)

        self.jacobian = 2 / (self.b-self.a)

        self.Vx = torch.zeros((self.M,self.M), dtype=torch.float64, device=self.device)
        for i in range(self.M):
            self.Vx[i,0] = 1
        
        for i in range(self.M):
            self.Vx[i,1] = self.z[i]
        
        for i in range(self.M):
            for j in range(2,self.M):
                self.Vx[i,j] = 2*self.z[i]*self.Vx[i,j-1]-self.Vx[i,j-2]
        

        d_inv = torch.full((M,), 2.0/M, dtype=torch.float64, device=device)
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

        matrix  = torch.zeros((self.M,self.M), dtype=torch.float64)

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


if __name__ == "__main__" :
    def test_identity(basis, M):
        for k in range(M):
            Tk = torch.cos(k * torch.arccos(torch.clamp((basis.z - (basis.a+basis.b)/2)/((basis.b-basis.a)/2), -1, 1)))
            c = basis.dbt(Tk)
            expected = torch.zeros(M, dtype=torch.float64)
            expected[k] = 1.0
            assert torch.allclose(c, expected, atol=1e-8), f"échec pour k={k}: {c}"
        print("OK: identité T_k -> one-hot")

    def test_roundtrip(basis, M):
        u = torch.randn(M, dtype=torch.float64)
        u_rec = basis.idbt(basis.dbt(u))
        assert torch.allclose(u, u_rec, atol=1e-8), (u - u_rec).abs().max()
        print("OK: round-trip")

    M = 8
    basis = TchebychevBasis1D(M, domain=(-1.0, 1.0))
    test_identity(basis, M)
    test_roundtrip(basis, M)
        
