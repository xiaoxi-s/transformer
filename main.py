import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F

from tqdm import tqdm
from matplotlib import pyplot as plt

from models.transformer import Transformer 
from data.utils import load_pickled_data, get_train_and_test_dataset
from hyperparams import *

import wandb

def train(model, train_loader, test_loader, criterion, optimizer, epochs=1):
    # Define loss function and optimizer
    # Training loop
    train_loss_history = []
    test_loss_history = []
    epoch_sequence = []

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')

    for epoch in range(epochs):
        # training
        log_registry = {}
        training_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            model.train()
            for batch in train_loader:
                inputs, labels = batch[:, 0, :].contiguous(), batch[:, 1, :].contiguous()
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
        train_loss_history.append(training_loss / len(train_loader.dataset))
        log_registry['train_loss'] = training_loss / len(train_loader.dataset)

        # testing
        test_loss = 0.0    
        with torch.no_grad():
            model.eval()
            for batch in test_loader:
                inputs, labels = batch[:, 0, :].contiguous(), batch[:, 1, :].contiguous()
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs, inputs)  # Forward pass: (B, T, Emb)
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                labels = labels.view(B * T)

                loss = criterion(logits, labels)  # Compute the loss
                test_loss += (loss.item() * len(batch))
        test_loss_history.append(test_loss / len(test_loader.dataset))
        log_registry['test_loss'] = test_loss / len(test_loader.dataset)
        wandb.log(log_registry)

        epoch_sequence.append(epoch + 1)

        # plt.plot(epoch_sequence, train_loss_history, 'b-', label='train loss')
        # plt.plot(epoch_sequence, test_loss_history, 'r-', label='test loss')
        # plt.show(block=False)

        # if epoch == 0:
        #     plt.legend()

        # plt.pause(0.001)

        if epoch > 1:
            print(f'Epoch {epoch + 1}/{epochs}: train loss {train_loss_history[-1]}, test loss {test_loss_history[-1]}')

        if epoch > 1 and epoch % 3 == 0:
            # Save the model
            torch.save(model.state_dict(), f'./data/model-{epoch}.pth')

    print('Training complete!')
    
    return model 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-e', '--epochs', default=20, type=int)           # positional argument
    parser.add_argument('-f', '--factor', default=0.0001, type=float)      # option that takes a value
    parser.add_argument('-p', '--parallel', default="true", type=str)      # option that takes a value
    parser.add_argument('-q', '--quiet-wandb', action="store_false")

    args = parser.parse_args()
    epochs = args.epochs 
    factor = args.factor
    quiet_wandb = args.quiet_wandb

    if quiet_wandb:
        print("Enable wandb")
        wandb.init(
            project="shakespear-transformer",
            config={
                "learning_rate": learning_rate, 
                "architecture": "Shakespear's transformer",
                "dataset": "Shakespear",
                "epochs": epochs,
                "factor": factor
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

    vocab_to_ind = load_pickled_data('vocab_to_ind.pkl') 

    model = Transformer(len(vocab_to_ind), dropout=dropout, block_size=block_size, num_of_decoder_layers=1, num_of_encoder_layers=1, dmodel=dmodel)
    if args.parallel.lower() == "true" or args.parallel.lower() == "t":
        print("Enable PyTorch Data parallelism")
        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        model = nn.DataParallel(model, device_ids=available_gpus)

    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    print("Token type number: ", len(vocab_to_ind))

    train_dataset, test_dataset = get_train_and_test_dataset(vocab_to_ind, factor=factor, device=device, block_size=block_size)

    print("Train dataset length: ", len(train_dataset))
    print("Test dataset length: ", len(test_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create an instance of your model
    model = train(model, train_loader, test_loader, criterion, optimizer, epochs=epochs)