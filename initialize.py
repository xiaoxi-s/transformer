import wandb
import torch

from data.dataset_utils import generate_dataset

from data.vocab_utils import get_vocab
from storage.local import LocalStorage
from storage.wandb import WandBStorage
from hyperparams import *


def initialize_torch():
    torch.manual_seed(7777)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)


def initialize_wandb(quiet_wandb, epochs, factor, tokenizer):
    if quiet_wandb:
        print("Enable wandb")
        wandb.init(
            project="shakespear-transformer",
            config={
                "learning_rate": learning_rate,
                "architecture": "Shakespear's transformer",
                "dataset": "Shakespear",
                "epochs": epochs,
                "factor": factor,
                "tokenizer": tokenizer,
            },
        )
    else:
        print("Disable wandb")
        wandb.init(mode="disabled")


def initialize_storage(
    location,
    wandb_project,
    model_artifact_name,
    dataset_artifact_name,
    vocab_artifact_name,
):
    if location == "local":
        print("Using local storage")
        storage = LocalStorage()
    elif location == "wandb":
        print("Using wandb storage")
        storage = WandBStorage(
            wandb_init_config={"project": wandb_project},
            model_artifact_name=model_artifact_name,
            dataset_artifact_name=dataset_artifact_name,
            vocab_artifact_name=vocab_artifact_name,
        )
    return storage


def intialize_vocab(storage, vocab_name, tokenizer, play_paths):
    print("Vocab storage: ")
    if storage.vocab_exists(vocab_name):
        print("  Existing vocab artifact found")
        vocab_to_ind = storage.load_vocab(vocab_name)
        print("  Vocab artifact loaded")
    else:
        print("  No existing vocab artifact found")
        print("  Generating vocab artifact")
        vocab_to_ind = get_vocab(tokenizer, play_paths)
        print("  Storing vocab artifact")
        storage.store_vocab(vocab_name, vocab_to_ind)
        print("  Vocab artifact stored")
    return vocab_to_ind


def initialize_dataset(
    dataset_name, storage, vocab_to_ind, play_paths, factor, block_size, tokenizer
):
    print("Dataset storage: ")
    if storage.dataset_exists(dataset_name):
        print("  Existing dataset artifact found")
        full_dataset = storage.load_dataset(dataset_name)
        print("  Dataset artifact loaded")
    else:
        print("  No existing dataset artifact found")
        print("  Generating dataset artifact")
        full_dataset = generate_dataset(
            vocab_to_ind,
            play_paths=play_paths,
            block_size=block_size,
            tokenizer=tokenizer,
        )
        print("  Storing dataset artifact")
        storage.store_dataset(dataset_name, full_dataset)
        print("  Dataset artifact stored")

    end_of_selected_data = int(len(full_dataset) * factor)
    full_dataset = full_dataset[0:end_of_selected_data]
    train_dataset, test_dataset, finetune_dataset, validation_dataset = (
        torch.utils.data.random_split(
            full_dataset, [0.7, 0.1498, 0.0004, 0.1498], torch.Generator(device=device)
        )
    )
    return train_dataset, test_dataset, finetune_dataset, validation_dataset
