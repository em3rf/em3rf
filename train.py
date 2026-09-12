from typing import List

import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.utilities.model_summary import ModelSummary

OmegaConf.register_new_resolver("getIndex", lambda lst, idx: lst[idx])


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def main(cfg: DictConfig):
    """
    Entry point for training the model.
    """
    if cfg.get("model") is None:
        raise ValueError(
            "Model configuration is missing, please specify a model to train."
        )

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    loggers: List[Logger] = [
        hydra.utils.instantiate(logger)
        for logger in cfg.get("loggers", dict()).values()
    ]

    # Log hyperparameters
    for logger in loggers:
        logger.log_hyperparams(OmegaConf.to_object(cfg))

    # Initialize the model
    model: L.LightningModule = hydra.utils.instantiate(cfg.get("model"))
    print(model)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.get("data"))
    callbacks: List[L.Callback] = [
        hydra.utils.instantiate(callback) for callback in cfg.get("callbacks").values()
    ]

    if cfg.get("ckpt_path") and cfg.get("finetuning"):
        state_dict = torch.load(cfg.get("ckpt_path"), map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
        model.enable_lora()

    # Initialize the trainer
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.get("trainer"), callbacks=callbacks, logger=loggers
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=(
            cfg.get("ckpt_path")
            if cfg.get("ckpt_path") and not cfg.get("finetuning")
            else None
        ),
    )


if __name__ == "__main__":
    main()
