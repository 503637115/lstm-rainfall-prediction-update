# train_deepspeed.py
import torch, deepspeed
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self):
        self.data = torch.rand(100, 20)
        self.labels = torch.rand(100)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self): super().__init__()
        self.lstm = nn.LSTM(1, 128, 2, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x): x = x.unsqueeze(-1)
        out, _ = self.lstm(x); return self.fc(out[:, -1, :])

model = LSTMModel()
parameters = filter(lambda p: p.requires_grad, model.parameters())
model, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=parameters,
    config={'train_batch_size': 16, 'optimizer': {'type': 'Adam', 'params': {'lr': 1e-4}}, 'fp16': {'enabled': False}})

train_loader = DataLoader(DummyDataset(), batch_size=16, shuffle=True)
loss_fn = nn.MSELoss()

for epoch in range(5):
    model.train()
    for x, y in train_loader:
        y_pred = model(x)
        loss = loss_fn(y_pred.squeeze(), y)
        model.backward(loss)
        model.step()
    print(f"Epoch {epoch+1} completed")
