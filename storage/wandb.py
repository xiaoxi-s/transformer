import torch
import torch.nn as nn
import os
import logging
import wandb

from typing import Any, Union
from storage.base_storage_backend import BaseStorageBackend
from storage.local import LocalStorage


class WandBStorage(BaseStorageBackend):
    """WandB storage backend for storing models, datasets, and vocab."""

    def __init__(
        self,
        wandb_init_config: dict,
        model_artifact_name: str,
        dataset_artifact_name: str,
        vocab_artifact_name: str,
        wandb_terminal_log_enabled: bool = False,
        model_path: str = "temp/model",
        dataset_path: str = "temp/dataset",
        vocab_path: str = "temp/vocab",
    ) -> None:
        super().__init__()

        # setup wandb logger
        logger = logging.getLogger("wandb")
        logger.propagate = wandb_terminal_log_enabled

        # artifact names
        self.model_artifact_name = model_artifact_name
        self.dataset_artifact_name = dataset_artifact_name
        self.vocab_artifact_name = vocab_artifact_name

        # setup storage paths
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.vocab_path = vocab_path
        self.local = LocalStorage(model_path, dataset_path, vocab_path)

        # initialize wandb
        self.wandb_init_config = wandb_init_config

    def store_model(
        self,
        name: str,
        artifact: Union[
            nn.Module, nn.DataParallel, nn.parallel.DistributedDataParallel
        ],
        *args: list,
        **kwargs: dict,
    ) -> None:
        self.local.store_model(name, artifact, *args, **kwargs)
        art = wandb.Artifact(self.model_artifact_name, type="model")
        model_path = os.path.join(self.model_path, name)
        art.add_file(model_path)
        wandb.log_artifact(art)

    def store_dataset(
        self,
        name: str,
        artifact: Any,
        *args: list,
        **kwargs: dict,
    ) -> None:
        self.local.store_dataset(name, artifact, *args, **kwargs)
        art = wandb.Artifact(self.dataset_artifact_name, type="dataset")
        dataset_path = os.path.join(self.dataset_path, name)
        art.add_file(dataset_path)
        wandb.log_artifact(art)

    def store_vocab(
        self,
        name: str,
        artifact: Any,
        *args: list,
        **kwargs: dict,
    ) -> None:
        self.local.store_vocab(name, artifact, *args, **kwargs)
        art = wandb.Artifact(
            self.vocab_artifact_name, type="dataset"
        )  # vocab is just a dict of Python
        vocab_path = os.path.join(self.vocab_path, name)
        art.add_file(vocab_path)
        wandb.log_artifact(art)

    def model_exists(self, name: str, *args: list, **kwargs: dict) -> bool:
        try:
            model_artifact = wandb.use_artifact(
                f"{self.model_artifact_name}:latest", type="model"
            )
            files = model_artifact.files()
            for file in files:
                if file.name == name:
                    return True
            return False
        except:
            return False

    def dataset_exists(self, name: str, *args: list, **kwargs: dict) -> bool:
        try:
            dataset_artifact = wandb.use_artifact(
                f"{self.dataset_artifact_name}:latest", type="dataset"
            )
            files = dataset_artifact.files()
            for file in files:
                if file.name == name:
                    return True
            return False
        except:
            return False

    def vocab_exists(self, name: str, *args: list, **kwargs: dict) -> bool:
        try:
            vocab_artifact = wandb.use_artifact(
                f"{self.vocab_artifact_name}:latest", type="dataset"
            )
            files = vocab_artifact.files()
            for file in files:
                if file.name == name:
                    return True
            return False
        except:
            return False

    def load_model(self, name, *args, **kwargs) -> Any:
        model_artifact = wandb.use_artifact(
            f"{self.model_artifact_name}:latest", type="model"
        )
        artifact_dir = model_artifact.download(root="./temp")
        model_path = os.path.join(artifact_dir, name)
        return torch.load(model_path)

    def load_dataset(self, name, *args, **kwargs) -> Any:
        dataset_artifact = wandb.use_artifact(
            f"{self.dataset_artifact_name}:latest", type="dataset"
        )
        artifact_dir = dataset_artifact.download(root="./temp")
        dataset_path = os.path.join(artifact_dir, name)
        return torch.load(dataset_path)

    def load_vocab(self, name, *args, **kwargs) -> Any:
        vocab_artifact = wandb.use_artifact(
            f"{self.vocab_artifact_name}:latest", type="dataset"
        )
        artifact_dir = vocab_artifact.download(root="./temp")
        vocab_path = os.path.join(artifact_dir, name)
        return torch.load(vocab_path)
