import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl


class MNISTModel(pl.LightningModule):
    def __init__(self, lr=1e-3, n_units_l1=128, n_units_l2=256):
        super().__init__()
        self.lr = lr
        self.layer_1 = nn.Linear(28 * 28, n_units_l1)
        self.layer_2 = nn.Linear(n_units_l1, n_units_l2)
        self.layer_3 = nn.Linear(n_units_l2, 10)

    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)