import argparse
import wandb
import pickle
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
    parser.add_argument('-d', '--dataset', default='default', type=str)

    args = parser.parse_args()
    epochs = args.epochs 
    factor = args.factor
    quiet_wandb = args.quiet_wandb
    tokenizer = args.tokenizer
    dataset = args.dataset

    print("Tokenizer: ")
    if tokenizer.lower() == 'char' and dataset == 'default':
        print("  Using char tokenizer")
        vocab_to_ind = load_pickled_data('char_vocab_to_ind.pkl') 
    elif tokenizer.lower() == 'char' and dataset != 'default':
        from os.path import isfile, join
        if not isfile('../data/pre_vocab_to_ind.pkl'):
            with open('./input.txt', 'r', encoding='utf-8') as f:
                text = f.read()
            chars = sorted(list(set(text)))
            vocab_to_ind = { ch:i for i,ch in enumerate(chars) }
            ind_to_vocab = { i:ch for i,ch in enumerate(chars) }
            with open('./data/pre_vocab_to_ind.pkl', 'wb') as outfile:
                pickle.dump(vocab_to_ind, outfile)
            with open('./data/ind_to_pre_vocab.pkl', 'wb') as outfile:
                pickle.dump(ind_to_vocab, outfile)
        else:
            vocab_to_ind = load_pickled_data('pre_vocab_to_ind.pkl') 
            ind_to_vocab = load_pickled_data('ind_to_pre_vocab.pkl') 

        encode = lambda s: [vocab_to_ind[c] for c in s] # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([ind_to_vocab[i] for i in l]) # decoder: take a list of integers, output a string
    elif tokenizer.lower() == 'word':
        print("  Using word tokenizer")
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

    print("CUDA setup")
    print("  CUDA available: ", torch.cuda.is_available())
    print("  CUDA device count: ", torch.cuda.device_count())
 
    print("Training spec")
    print("  Epochs: ", epochs)
    print("  Learning rate: ", learning_rate)
    print("  Batch size: ", batch_size)
    print("  Drop out: ", dropout)

    torch.manual_seed(7777)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)

    num_of_decoder_layers=4
    num_of_encoder_layers=4
    model = Transformer(len(vocab_to_ind), dropout=dropout, block_size=block_size, num_of_decoder_layers=4, num_of_encoder_layers=4, dmodel=dmodel)
    if args.parallel.lower() == "true" or args.parallel.lower() == "t":
        print("Enable PyTorch Data parallelism")
        available_gpus = [i for i in range(torch.cuda.device_count())]
        model = nn.DataParallel(model, device_ids=available_gpus)

    print("Transformer spec")
    print("  Embedding dim: ", dmodel)
    print("  Max context length: ", block_size)
    print(f"  Number of decoder: {num_of_decoder_layers} - Number of encoder: {num_of_decoder_layers}")
    print("  Total num of model params: ", sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    train_dataset, test_dataset, finetune_dataset, validation_dataset = get_train_and_test_dataset(vocab_to_ind, dataset=dataset, factor=factor, device=device, block_size=block_size, tokenizer=tokenizer)

    print("Data spec")
    print("  Data factor (proportion of all Shakespeare's plays): ", args.factor)
    print("  Token type number: ", len(vocab_to_ind))
    print("  Train dataset length: ", len(train_dataset))
    print("  Test dataset length: ", len(test_dataset))
    print("  Finetune dataset length: ", len(finetune_dataset))
    print("  Validation dataset length: ", len(validation_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=finetune_batch_size, shuffle=True, generator=torch.Generator(device=device))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=finetune_batch_size, shuffle=True, generator=torch.Generator(device=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create an instance of your model
    model = train(model, tokenizer, train_loader, test_loader, criterion, optimizer, epochs=epochs)