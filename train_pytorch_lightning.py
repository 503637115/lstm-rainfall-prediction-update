# train_lightning.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

class DummyDataset(Dataset):
    def __init__(self): self.data = torch.rand(100, 20); self.labels = torch.rand(100)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

class LSTMModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 128, 2, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x): x = x.unsqueeze(-1)
        out, _ = self.lstm(x); return self.fc(out[:, -1, :])

    def training_step(self, batch, batch_idx):
        x, y = batch; y_pred = self(x)
        loss = nn.MSELoss()(y_pred.squeeze(), y)
        return loss

    def configure_optimizers(self): return optim.Adam(self.parameters(), lr=1e-4)

model = LSTMModel()
train_loader = DataLoader(DummyDataset(), batch_size=16, shuffle=True)
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_loader)
