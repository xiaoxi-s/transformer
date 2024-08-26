import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from os import listdir
from os.path import isfile, join

from models.transformer import Transformer 
from data.utils import load_pickled_data, get_train_and_test_dataset
from hyperparams import *
from train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='shakespear-training',
                    description='pretrain shakespeare transformer')
    parser.add_argument('-e', '--epochs', default=20, type=int)           # positional argument
    parser.add_argument('-f', '--factor', default=0.0001, type=float)      # option that takes a value
    parser.add_argument('-p', '--parallel', default="true", type=str)      # option that takes a value
    parser.add_argument('-q', '--quiet-wandb', action="store_false")
    parser.add_argument('-t', '--tokenizer', default='char', type=str)           # positional argument
    parser.add_argument('-d', '--dataset-name', default='default', type=str)           # positional argument

    args = parser.parse_args()
    epochs = args.epochs 
    factor = args.factor
    quiet_wandb = args.quiet_wandb
    tokenizer = args.tokenizer
    dataset_name = args.dataset_name

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
                "batch_size": batch_size,
                "block_size": block_size,
                "dmodel": dmodel,
                "dropout": dropout,
            }
        )
    else:
        print("Disable wandb")
        wandb.init(mode="disabled")

    print("Hello World!")
    print("CUDA available: ", torch.cuda.is_available())
    print("CUDA device count: ", torch.cuda.device_count())
    print("Epochs: ", epochs)
    print("Data factor: ", args.factor)
    torch.manual_seed(7777)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)

    vocab_path = f"{tokenizer}_vocab_to_ind_on_{dataset_name}.pkl"
    vocab_to_ind = load_pickled_data(vocab_path)

    if dataset_name == 'default':
        dataset_path = './shakespeare/shakespeare-db/'
    elif dataset_name == 'preprocessed':
        dataset_path = './input.txt'
    if isfile(dataset_path):
        plays = [dataset_path]
    else:
        plays = [join(dataset_path, f) for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

    model = Transformer(len(vocab_to_ind), dropout=dropout, block_size=block_size, num_of_decoder_layers=num_of_decoder_layers, num_of_encoder_layers=num_of_encoder_layers, dmodel=dmodel)
    if args.parallel.lower() == "true" or args.parallel.lower() == "t":
        print("Enable PyTorch Data parallelism")
        available_gpus = [i for i in range(torch.cuda.device_count())]
        model = nn.DataParallel(model, device_ids=available_gpus)

    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    print("Token type number: ", len(vocab_to_ind))

    train_dataset, test_dataset, finetune_dataset, validation_dataset = get_train_and_test_dataset(vocab_to_ind, dataset_name, plays, tokenizer=tokenizer, factor=factor, device=device, block_size=block_size)

    print("Hyper Parameters: ")
    print("  Learning rate: ", learning_rate)
    print("  Batch size: ", batch_size)
    print("  Block size: ", block_size)
    print("  Dmodel: ", dmodel)
    print("  Dropout: ", dropout)
    print("  Number of epochs: ", epochs)

    print("Data Spec: ")
    print("  Train dataset length: ", len(train_dataset))
    print("  Test dataset length: ", len(test_dataset))
    print("  Finetune dataset length: ", len(finetune_dataset))
    print("  Validation dataset length: ", len(validation_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create an instance of your model
    model = train(model, train_loader, test_loader, criterion, optimizer, epochs=epochs)