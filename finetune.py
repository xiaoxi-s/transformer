import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from models.transformer import Transformer 
from tqdm import tqdm
from data.utils import get_train_and_test_dataset, load_pickled_data
from hyperparams import *

def finetune(model, finetune_dataset, criterion, optimizer, finetune_block_size, original_name, epochs=1):
    finetune_history = []

    for epoch in range(epochs):
        # training
        log_registry = {}
        training_loss = 0.0
        with tqdm(total=len(finetune_dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            model.train()
            for batch in finetune_dataset:
                print("hello!!", batch.size() )
                inputs, labels = batch[:, 0, 0:finetune_block_size].contiguous(), batch[:, 1, 0:finetune_block_size].contiguous()
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # Zero the gradients
                logits = model(inputs, inputs)  # Forward pass: (B, T, Emb)
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                labels = labels.view(B * T)

                loss = criterion(logits, labels)  # Compute the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                training_loss += (loss.item() * len(batch))
                pbar.update(1)  # Update the progress bar
        finetune_history.append(training_loss / len(finetune_dataset.dataset))
        log_registry['finetune'] = training_loss / len(finetune_dataset.dataset)

        wandb.log(log_registry)
        torch.save(model.state_dict(), f'./data/{original_name}-finetuned-{epoch}.pth')

    return model

if __name__ == "__main__":
    torch.manual_seed(7777)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)
    np.random.seed(7777)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--name-of-finetune-model', type=str) 
    argparser.add_argument('-e', '--finetune-epoch', type=int, default=5)
    argparser.add_argument('-f', '--factor', default=0.0001, type=float)      # option that takes a value
    argparser.add_argument('-p', '--parallel', default="true", type=str)      # option that takes a value
    argparser.add_argument('-q', '--quiet-wandb', action="store_false")

    parser = argparser.parse_args()
    finetune_epoch = parser.finetune_epoch
    name_of_finetune_model = parser.name_of_finetune_model
    parallel = parser.parallel
    factor = parser.factor

    quiet_wandb = parser.quiet_wandb

    if quiet_wandb:
        print("Enable wandb")
        wandb.init(
            project="shakespear-transformer",
            config={
                "learning_rate": finetune_learning_rate, 
                "architecture": "Shakespear's transformer",
                "dataset": "Shakespear",
                "finetune_epochs": finetune_epoch,
                "factor": factor
            }
        )
    else:
        print("Disable wandb")
        wandb.init(mode="disabled")

    print("Loading vocab ...")
    vocab_to_ind = load_pickled_data('vocab_to_ind.pkl') 
    ind_to_vocab = load_pickled_data('ind_to_vocab.pkl')
    print("Done loading vocab ...")
    torch.set_default_device(device)
    print("Vocab size: ", len(vocab_to_ind))
    _, _, finetune_dataset, _ = get_train_and_test_dataset(vocab_to_ind, factor=factor, device=device, block_size=block_size)
    finetune_loader = torch.utils.data.DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

    model = Transformer(len(vocab_to_ind), dropout=dropout, block_size=block_size, num_of_decoder_layers=1, num_of_encoder_layers=1, dmodel=dmodel).to(device) 
    if parser.parallel.lower() == "true" or parser.parallel.lower() == "t":
        model = nn.DataParallel(Transformer(len(vocab_to_ind), dropout=dropout, block_size=block_size, num_of_decoder_layers=1, num_of_encoder_layers=1, dmodel=dmodel).to(device)) 
    
    model.load_state_dict(torch.load(f'data/{name_of_finetune_model}'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=finetune_learning_rate)
    finetune(model, test_loader, criterion, optimizer, finetune_block_size, name_of_finetune_model, finetune_epoch)