import torch

def eval_chunked(model, *coords, n_chunks=10):

    coords_chunked = [torch.chunk(x, n_chunks, dim=0) for x in coords]
    u_pred = []
    for i in range(n_chunks):
        coord_chunk = [x[i] for x in coords_chunked]
        with torch.no_grad():
            u_chunk = model(*coord_chunk)
        u_pred.append(u_chunk)
    u = torch.cat(u_pred, dim=0)
    return u
