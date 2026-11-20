import numpy as np
import torch
from torch import nn


def conv(inputs, outputs, bias=True):
    return nn.Conv2d(inputs, outputs, 5, padding=2, padding_mode="circular", bias=bias)

def mapping(channels):
    return nn.Sequential(conv(channels, channels), nn.Tanh(), conv(channels, channels))

class EFNN(nn.Module):
    def __init__(self, channels=3, layers=3):
        super().__init__()
        self.left = nn.ModuleList()
        self.right = nn.ModuleList()
        self.f = nn.ModuleList()
        self.g = nn.ModuleList()
        for i in range(layers + 1):
            self.left.append(conv(1, channels, bias=False))
            self.right.append(conv(1, channels, bias=False))
        for i in range(layers):
            self.f.append(mapping(channels))
            self.g.append(mapping(channels))
        self.output = conv(channels, 1)

    def symmetrize(self, spins, i):
        result = 0
        for k in range(3):
            component = spins[:, k:k+1]
            result = result + self.left[i](component) * self.right[i](component)
        return result

    def forward(self, spins):
        state = self.symmetrize(spins, 0)
        for i in range(len(self.f)):
            field = self.f[i](state)
            state = field * self.g[i](self.symmetrize(spins, i+1))
        return self.output(state).sum(dim=(1, 2, 3))


def energy(spins, t=1.0, J=16.0, mu=-8.3, T=0.1):
    N = spins.shape[-1]
    sites = np.arange(N*N).reshape(N, N)
    hopping = np.zeros((N*N, N*N))
    for axis in (0, 1):
        neighbors = np.roll(sites, -1, axis=axis).ravel()
        hopping[sites.ravel(), neighbors] = -t
        hopping[neighbors, sites.ravel()] = -t
    x, y, z = spins.reshape(3, -1)
    up = hopping + np.diag(-mu-J*z/2)
    down = hopping + np.diag(-mu+J*z/2)
    flip = np.diag(-J*(x-1j*y)/2)
    H = np.block([[up, flip], [flip.conj().T,down]])
    eigenvalues = np.linalg.eigvalsh(H)
    return -T * np.logaddexp(0, -eigenvalues/T).sum()
