import math
import numpy as np
import torch
import torch.nn.functional as F

def skew_symmetric_matrix(n, batch_size=1):
    A = torch.randn(batch_size, n, n)
    skew_symm = (A - A.transpose(-1, -2))
    return skew_symm

def psd_matrix(n, batch_size=1):
    A = torch.randn(batch_size, n, n)
    psd = A @ A.transpose(-1, -2) / (math.sqrt(n))
    return psd

def diag_matrix(n, batch_size=1):
    diag = torch.distributions.LogNormal(0, 1).sample((batch_size, n))
    return torch.diag_embed(diag)

def laplacian_matrix(n, batch_size=1):
    lap_q = torch.randn(batch_size, n, n)
    lap_k = torch.randn(batch_size, n, n)
    scores = lap_q @ lap_k.transpose(-1, -2) / math.sqrt(n)
    scores = 0.5 * (scores + scores.transpose(-1, -2))
    adjacency = F.softplus(scores)
    adjacency = adjacency - torch.diag_embed(
        torch.diagonal(adjacency, dim1=-2, dim2=-1)
    )
    degree = torch.diag_embed(adjacency.sum(dim=-1))
    laplacian = degree - adjacency
    return laplacian


if __name__ == "__main__":
    n = 10
    batch_size = 5

    skew_symm = skew_symmetric_matrix(n, batch_size)
    psd = psd_matrix(n, batch_size)
    diag = diag_matrix(n, batch_size)
    laplacian = laplacian_matrix(n, batch_size)

    print("Skew-symmetric matrix eigenvalues:")
    print(torch.linalg.eigvals(skew_symm))

    print("\nPSD matrix eigenvalues:")
    print(torch.linalg.eigvals(psd))

    print("\nDiagonal matrix eigenvalues:")
    print(torch.linalg.eigvals(diag))

    print("\nLaplacian matrix eigenvalues:")
    print(laplacian)
    print(torch.linalg.eigvals(laplacian))