from abc import ABC, abstractmethod

from typing import Any

class BaseStorageBackend(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def store_model(self, name, artifact, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def store_dataset(self, name, artifact, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def store_vocab(self, name, artifact, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def model_exists(self, name, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    def dataset_exists(self, name, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    def vocab_exists(self, name, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    def load_model(self, name, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def load_dataset(self, name, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def load_vocab(self, name, *args, **kwargs) -> Any:
        pass

