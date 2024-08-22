import torch
import torch.nn as nn
import os

from typing import Any, Union 
from storage.base_storage_backend import BaseStorageBackend

class LocalStorage(BaseStorageBackend):
    """Local storage backend for storing models, datasets, and vocab."""
    def __init__(self, model_path: str='data/model', dataset_path: str='data/dataset', vocab_path: str='data/vocab') -> None:
        super().__init__()
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.vocab_path = vocab_path

    def store_model(self, name: str, artifact: Union[nn.Module, nn.DataParallel, nn.parallel.DistributedDataParallel], *args: list, **kwargs: dict) -> None:
        storage_path = os.path.join(self.model_path, name)
        if not os.path.exists(os.path.dirname(storage_path)):
            os.makedirs(os.path.dirname(storage_path))
        torch.save(artifact.state_dict(), storage_path)

    def store_dataset(self, name: str, artifact: torch.Tensor, *args: list, **kwargs: dict) -> None:
        storage_path = os.path.join(self.dataset_path, name)
        if not os.path.exists(os.path.dirname(storage_path)):
            os.makedirs(os.path.dirname(storage_path))
        torch.save(artifact, storage_path) 

    def store_vocab(self, name: str, artifact: dict, *args: list, **kwargs: dict) -> None:
        storage_path = os.path.join(self.vocab_path, name)
        if not os.path.exists(os.path.dirname(storage_path)):
            os.makedirs(os.path.dirname(storage_path))
        torch.save(artifact, storage_path)
    
    def model_exists(self, name: str, *args: list, **kwargs: dict) -> bool:
        storage_path = os.path.join(self.model_path, name)
        return os.path.exists(storage_path)

    def dataset_exists(self, name: str, *args: list, **kwargs: dict) -> bool:
        storage_path = os.path.join(self.dataset_path, name)
        return os.path.exists(storage_path)

    def vocab_exists(self, name: str, *args: list, **kwargs: dict) -> bool:
        vocab_path = os.path.join(self.vocab_path, name)
        return os.path.exists(vocab_path)

    def load_model(self, name: str, *args: list, **kwargs: dict) ->Any:
        return torch.load(os.path.join(self.model_path, name))

    def load_dataset(self, name: str, *args: list, **kwargs: dict) -> Any:
        return torch.load(os.path.join(self.dataset_path, name))

    def load_vocab(self, name: str, *args: list, **kwargs: dict) -> Any:
        return torch.load(os.path.join(self.vocab_path, name))