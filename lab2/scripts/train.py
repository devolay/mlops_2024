import os
import torch
import optuna

import torch.nn.functional as F
import lightning.pytorch as pl

from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import NeptuneLogger

from dataset import MNISTDataModule
from model import MNISTModel

load_dotenv()


def main():
    api_token = os.getenv("NEPTUNE_API_TOKEN")
    project_name = os.getenv("NEPTUNE_PROJECT")

    neptune_logger = NeptuneLogger(
        api_key=api_token,
        project=project_name,
        name="MNIST-training",
    )

    dataset = MNISTDataModule()
    model = MNISTModel()

    trainer = pl.Trainer(
        max_epochs=5,
        logger=neptune_logger,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
    )
    trainer.fit(model, dataset)


if __name__ == "__main__":
    main()
