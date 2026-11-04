import json
import time
from autoencoder.measurement import configure_affinity
configure_affinity()
import torch
from autoencoder.model import Autoencoder, synthetic

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(42)
train = synthetic(64, 120)
mean, std = train.mean((0, 1)), train.std((0, 1)).clamp_min(1e-6)
train = (train - mean) / std
heldout = (synthetic(16, 120, 456) - mean) / std
model = Autoencoder()
assert sum(p.numel() for p in model.parameters()) == 1003952
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
with torch.no_grad():
    before = (model(heldout) - heldout).square().mean().item()
losses = []
start = time.perf_counter()
for step in range(40):
    x = train[(step * 4) % 64: (step * 4) % 64 + 4]
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - x).square().mean()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
model.eval()
with torch.inference_mode():
    after = (model(heldout) - heldout).square().mean().item()
torch.save(dict(state_dict=model.state_dict(), mean=mean, std=std), "artifacts/model.pt")
result = dict(seed=42, training_shape=[64,120,32], steps=40, batch=4,
              parameters=sum(p.numel() for p in model.parameters()),
              seconds=time.perf_counter()-start, heldout_mse_before=before,
              heldout_mse_after=after, losses=losses)
open("artifacts/training.json", "w").write(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k != "losses"}))
