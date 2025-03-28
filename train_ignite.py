# train_ignite.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError

class DummyDataset(Dataset):
    def __init__(self): self.data = torch.rand(100, 20); self.labels = torch.rand(100)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 128, 2, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x): x = x.unsqueeze(-1)
        out, _ = self.lstm(x); return self.fc(out[:, -1, :])

model = LSTMModel()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def train_step(engine, batch):
    model.train(); optimizer.zero_grad()
    x, y = batch[0].to(device), batch[1].to(device)
    y_pred = model(x); loss = loss_fn(y_pred.squeeze(), y)
    loss.backward(); optimizer.step(); return loss.item()

trainer = Engine(train_step)
data_loader = DataLoader(DummyDataset(), batch_size=16, shuffle=True)
MeanSquaredError().attach(trainer, "mse")

@trainer.on(Events.EPOCH_COMPLETED)
def log_metrics(engine): print(f"Epoch {engine.state.epoch}: MSE = {engine.state.metrics['mse']:.4f}")

trainer.run(data_loader, max_epochs=5)
