import argparse
import torch
import torch.nn as nn
import torch.optim as optim


from models.transformer import Transformer
from data.utils import get_play_paths
from hyperparams import *
from constants import *
from train import train
from initialize import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="shakespear-training", description="pretrain shakespeare transformer"
    )
    parser.add_argument("-e", "--epochs", default=20, type=int)  # positional argument
    parser.add_argument(
        "-f", "--factor", default=0.0001, type=float
    )  # option that takes a value
    parser.add_argument(
        "-p", "--parallel", default="true", type=str
    )  # option that takes a value
    parser.add_argument("-q", "--quiet-wandb", action="store_false")
    parser.add_argument("-t", "--tokenizer", default="char", type=str)
    parser.add_argument("-d", "--dataset", default="default", type=str)
    parser.add_argument("-l", "--location", default="local", type=str)

    args = parser.parse_args()
    epochs = args.epochs
    factor = args.factor
    quiet_wandb = args.quiet_wandb
    tokenizer = args.tokenizer
    dataset = args.dataset
    location = args.location
    data_parallel_enabled = (
        args.parallel.lower() == "true" or args.parallel.lower() == "t"
    )

    dataset_name = f"dataset-{dataset}-by-tokenizer-{tokenizer}.pth"
    vocab_name = f"vocab-{tokenizer}-for-dataset-{dataset}.pth"

    # Find the proper dataset
    if dataset == "default":
        dataset_path = "shakespeare/shakespeare-db"
    elif dataset == "preprocessed":
        dataset_path = "./input.txt"
    else:
        raise ValueError("Invalid dataset. Can only be default or preprocessed.")
    play_paths = get_play_paths(dataset_path)

    # Initialize the vocab, dataset, and storage
    initialize_torch()
    initialize_wandb(quiet_wandb, epochs, factor, tokenizer)
    storage = initialize_storage(
        location,
        wandb_project,
        model_artifact_name,
        dataset_artifact_name,
        vocab_artifact_name,
    )  # these parameters are constants for the project
    vocab_to_ind = intialize_vocab(storage, vocab_name, tokenizer, play_paths)
    train_dataset, test_dataset, finetune_dataset, validation_dataset = (
        initialize_dataset(
            dataset_name,
            storage,
            vocab_to_ind,
            play_paths,
            factor,
            block_size,
            tokenizer,
        )
    )

    print("Data spec")
    print("  Data factor (proportion of all Shakespeare's plays): ", args.factor)
    print("  Token type number: ", len(vocab_to_ind))
    print("  Train dataset length: ", len(train_dataset))
    print("  Test dataset length: ", len(test_dataset))
    print("  Finetune dataset length: ", len(finetune_dataset))
    print("  Validation dataset length: ", len(validation_dataset))

    num_of_decoder_layers = 4
    num_of_encoder_layers = 4
    model = Transformer(
        len(vocab_to_ind),
        dropout=dropout,
        block_size=block_size,
        num_of_decoder_layers=num_of_decoder_layers,
        num_of_encoder_layers=num_of_encoder_layers,
        dmodel=dmodel,
    )
    if data_parallel_enabled:
        available_gpus = [i for i in range(torch.cuda.device_count())]
        model = nn.DataParallel(model, device_ids=available_gpus)

    print("Transformer spec")
    print("  Embedding dim: ", dmodel)
    print("  Max context length: ", block_size)
    print(
        f"  Number of decoder: {num_of_decoder_layers} - Number of encoder: {num_of_decoder_layers}"
    )
    print(
        "  Total num of model params: ",
        sum(p.numel() for p in model.parameters()) / 1e6,
        "M parameters",
    )
    print("  Data parallelism enabeld: ", data_parallel_enabled)

    print("CUDA setup")
    print("  CUDA available: ", torch.cuda.is_available())
    print("  CUDA device count: ", torch.cuda.device_count())

    print("Training spec")
    print("  Epochs: ", epochs)
    print("  Learning rate: ", learning_rate)
    print("  Batch size: ", batch_size)
    print("  Drop out: ", dropout)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device=device),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device=device),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create an instance of your model
    model_name_prefix = f"model-on-{dataset}-with-{tokenizer}"
    model = train(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        epochs,
        model_name_prefix,
        storage,
    )
