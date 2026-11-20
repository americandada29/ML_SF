import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import EFNN, energy
import matplotlib.pyplot as plt
from tqdm import tqdm

N = 16
samples = 500
channels = 1
layers = 1
epochs = 1000
batch_size = 1000
t = 1.0
J = 16.0
mu = -8.3
T = 0.1


torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}. Generating {samples} exact-diagonalization labels.")
training, testing = torch.load("sample_data.pt", weights_only=False)
testing = DataLoader(testing, batch_size=batch_size)
model = EFNN(channels, layers).to(device)
model.load_state_dict(torch.load("efnn.pt", map_location=device, weights_only=True))
model.eval()
squared_error = 0.0

outs = []
with torch.no_grad():
    for i in tqdm(range(samples)):
        theta = np.random.uniform(0, np.pi, (N, N))
        phi = 2 * theta
        spin = np.stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)))
        x = torch.tensor(spin, dtype=torch.float32, device=device).unsqueeze(0)
        pred = model(x)
        y = energy(spin, t, J, mu, T)
        outs.append([pred.item(), y])

outs = np.array(outs)
plt.scatter(outs[:, 0], outs[:, 1])
plt.plot([np.amin(outs[:, 0]), np.amax(outs[:, 0])], [np.amin(outs[:, 0]), np.amax(outs[:, 0])], linestyle="--", c="black")
plt.show()


