import os
import torch
import optuna

import torch.nn.functional as F
import lightning.pytorch as pl

from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import NeptuneLogger
from optuna.integration import PyTorchLightningPruningCallback

from dataset import MNISTDataModule
from model import MNISTModel

load_dotenv()


def main():
    api_token = os.getenv("NEPTUNE_API_TOKEN")
    project_name = os.getenv("NEPTUNE_PROJECT")

    def objective(trial):
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        n_units_l1 = trial.suggest_int("n_units_l1", 64, 256)
        n_units_l2 = trial.suggest_int("n_units_l2", 64, 256)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        dataset = MNISTDataModule(batch_size=batch_size)
        model = MNISTModel(lr=lr, n_units_l1=n_units_l1, n_units_l2=n_units_l2)

        neptune_logger = NeptuneLogger(
            api_key=api_token,
            project=project_name,
            name=f"MNIST-training-trial-{trial.number}",
        )

        trainer = pl.Trainer(
            max_epochs=5,
            logger=neptune_logger,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=3),
                PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            ],
        )
        trainer.fit(model, dataset)

        return trainer.callback_metrics["val_loss"].item()

    print("Running the optimization with Optuna")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
