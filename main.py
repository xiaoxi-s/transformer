import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F

from tqdm import tqdm
from matplotlib import pyplot as plt

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
    parser.add_argument('-t', '--tokenizer', default='char', type=str)

    args = parser.parse_args()
    epochs = args.epochs 
    factor = args.factor
    quiet_wandb = args.quiet_wandb
    tokenizer = args.tokenizer

    if tokenizer.lower() == 'char':
        print("Using char tokenizer")
        vocab_to_ind = load_pickled_data('char_vocab_to_ind.pkl') 
    elif tokenizer.lower() == 'word':
        print("Using word tokenizer")
        vocab_to_ind = load_pickled_data('vocab_to_ind.pkl') 
    else:
        raise ValueError("Invalid tokenizer. Can only be char or word.")

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
                "tokenizer": tokenizer
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

    model = Transformer(len(vocab_to_ind), dropout=dropout, block_size=block_size, num_of_decoder_layers=4, num_of_encoder_layers=4, dmodel=dmodel)
    if args.parallel.lower() == "true" or args.parallel.lower() == "t":
        print("Enable PyTorch Data parallelism")
        available_gpus = [i for i in range(torch.cuda.device_count())]
        model = nn.DataParallel(model, device_ids=available_gpus)

    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    print("Token type number: ", len(vocab_to_ind))

    train_dataset, test_dataset, finetune_dataset, validation_dataset = get_train_and_test_dataset(vocab_to_ind, factor=factor, device=device, block_size=block_size, tokenizer=tokenizer)

    print("Train dataset length: ", len(train_dataset))
    print("Test dataset length: ", len(test_dataset))
    print("Finetune dataset length: ", len(finetune_dataset))
    print("Validation dataset length: ", len(validation_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=finetune_batch_size, shuffle=True, generator=torch.Generator(device=device))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=finetune_batch_size, shuffle=True, generator=torch.Generator(device=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create an instance of your model
    model = train(model, tokenizer, train_loader, test_loader, criterion, optimizer, epochs=epochs)