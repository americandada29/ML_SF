"""Run with: python train.py. Requires numpy and torch.

Paper: https://arxiv.org/abs/2502.17665v2
Code: https://github.com/saschapojot/deep_field_collection/tree/main/final_sum_qt_efnn
Uses the paper's 2 million samples and 80/20 split. The released training
script instead names a 200,000-sample file; its optimizer settings are used here.
For a quick trial, reduce samples, epochs, and batch_size below.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import EFNN, energy
from tqdm import tqdm

N = 5
samples = 2**16
channels = 1
layers = 1
epochs = 300
batch_size = 1024
t = 1.0
J = 16.0
mu = -8.3
T = 0.1


torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}. Generating {samples} exact-diagonalization labels.")
spins = torch.empty(samples, 3, N, N)
energies = torch.empty(samples)
for i in tqdm(range(samples)):
    rng = np.random.RandomState(100 + i)
    theta = rng.uniform(0, np.pi, (N, N))
    phi = 2 * theta
    spin = np.stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)))
    spins[i] = torch.from_numpy(spin)
    energies[i] = energy(spin, t, J, mu, T)


split = int(0.8*samples)
training = TensorDataset(spins[:split], energies[:split])
testing = TensorDataset(spins[split:], energies[split:])
training = DataLoader(training, batch_size=batch_size, shuffle=True)

torch.save((training, testing), "sample_data.pt")

testing = DataLoader(testing, batch_size=batch_size)
model = EFNN(channels, layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
gamma = 0.1**(1/max(1, epochs//50))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=gamma)

for epoch in range(epochs):
    model.train()
    squared_error = 0.0
    for x, y in training:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        squared_error += loss.item() * len(y)
    scheduler.step()
    print(f"Epoch {epoch + 1}/{epochs}: train RMSE = {(squared_error / split) ** 0.5:.6f}")
torch.save(model.cpu().state_dict(), "efnn.pt")


# model.eval()
# squared_error = 0.0
# with torch.no_grad():
#     for x, y in testing:
#         x, y = x.to(device), y.to(device)
#         squared_error += ((model(x) - y) ** 2).sum().item()
# rmse = (squared_error / (samples - split)) ** 0.5
# relative_error = rmse / abs(energies[split:].mean().item())
# print(f"Test RMSE = {rmse:.6f}, relative error = {relative_error:.6f}")




