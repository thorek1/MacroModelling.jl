
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Enable CPU threading
torch.set_num_threads(8)  # Adjust based on your CPU cores


with h5py.File("/Users/thorekockerols/GitHub/MacroModelling.jl/data.h5", "r") as f:
    inputs = f["inputs"][:]
    outputs = f["outputs"][:]

# Convert to tensors with memory pinning
input_tensor = torch.tensor(inputs)#.pin_memory()
output_tensor = torch.tensor(outputs)#.pin_memory()

# Define the model
model = nn.Sequential(
    nn.Linear(input_tensor.shape[1], 256),
    nn.LeakyReLU(),
    nn.Linear(256, 256),
    # nn.Tanh(),
    # nn.Linear(128, 128),
    # nn.LeakyReLU(),
    # nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(256, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, output_tensor.shape[1])
)



epochs = 300

# Define loss function and optimizer
criterion = lambda output, target: torch.sqrt(nn.MSELoss()(output, target))
optimizer = optim.AdamW(model.parameters())
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min = 1e-10, T_max=epochs)

dataset = TensorDataset(input_tensor, output_tensor)
dataloader = DataLoader(
    dataset,
    batch_size=1024,  # Experiment with batch size
    shuffle=True,
    num_workers=4,   # Adjust based on CPU cores
    # pin_memory=True
)


# Training loop
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_inputs, batch_outputs in dataloader:
        optimizer.zero_grad()
        outputs_pred = model(batch_inputs)
        loss = criterion(outputs_pred, batch_outputs)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}, Learning Rate: {optimizer.param_groups[0]["lr"]}')


