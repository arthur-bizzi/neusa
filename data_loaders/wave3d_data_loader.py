import torch
from basis.cosine3d import CosineBasis3D

def wave3d_groundtruth_generator():

    # domain
    domain_x = [-1.5,1.5]
    domain_y = [-1.5,1.5]
    domain_z = [-1.5,1.5]
    domain_t = [0.0,1.0]

    # resolution
    n_x = 101
    n_y = 101
    n_z = 101
    n_t = 111

    # basis and grid
    basis = CosineBasis3D(n_x,n_y,n_z,domain_x,domain_y,domain_z)
    X,Y,Z = torch.meshgrid(basis.x,basis.y,basis.z, indexing='ij')
    grid = torch.stack((X,Y,Z),dim=-1)
    xs = grid[:,:,:,0:1]
    ys = grid[:,:,:,1:2]
    zs = grid[:,:,:,2:3]

    ts = torch.linspace(*domain_t,n_t)

    # initial condition
    sigma = 0.15
    u0 = 10*torch.exp((-xs**2-ys**2-zs**2)/(2*sigma**2)).squeeze(-1)
    v0 = torch.zeros_like(u0)

    # velocity field
    c = (1.0 - 0.5 * (torch.sigmoid(1000 * (zs - 0.5)))).squeeze(-1)

    def rk4(u, v, dt, f, g):
        k1 = dt * f(u, v)
        l1 = dt * g(u, v)

        k2 = dt * f(u + 0.5 * k1, v + 0.5 * l1)
        l2 = dt * g(u + 0.5 * k1, v + 0.5 * l1)

        k3 = dt * f(u + 0.5 * k2, v + 0.5 * l2)
        l3 = dt * g(u + 0.5 * k2, v + 0.5 * l2)

        k4 = dt * f(u + k3, v + l3)
        l4 = dt * g(u + k3, v + l3)

        u_next = u + (k1 + 2*k2 + 2*k3 + k4) / 6
        v_next = v + (l1 + 2*l2 + 2*l3 + l4) / 6

        return u_next, v_next

    def fu(_, v):

        return v

    def fv(u, _):
        u_hat = basis.dbt(u)

        D2u_hat = basis.D2 * u_hat
        D2u = basis.idbt(D2u_hat)
       
        return c**2 * D2u

    T = [0.0]
    U = [u0]
    V = [v0]
    u = u0
    v = v0
    dt = ts[1] - ts[0]
    for t in ts[1:]:
        u,v = rk4(u,v,dt,fu,fv)

        T.append(t)
        U.append(u)
        V.append(v)

    U_tensor = torch.stack(U)
    u_gt = U_tensor.detach().cpu().numpy()

    return u_gt
    