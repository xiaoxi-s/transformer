
import torch
import wandb

from tqdm import tqdm
from hyperparams import *


def train(model, train_loader, test_loader, criterion, optimizer, epochs=1):
    # Define loss function and optimizer
    # Training loop
    train_loss_history = []
    test_loss_history = []
    epoch_sequence = []

    for epoch in range(epochs):
        # training
        log_registry = {}
        training_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            model.train()
            for batch in train_loader:
                inputs, labels = batch[0].contiguous(), batch[1].contiguous()
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
                inputs, labels = batch[0].contiguous(), batch[1].contiguous()
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

        if epoch > 1:
            print(f'Epoch {epoch + 1}/{epochs}: train loss {train_loss_history[-1]}, test loss {test_loss_history[-1]}')

        if epoch > 1 and epoch % 3 == 0:
            # Save the model
            torch.save(model.state_dict(), f'./data/model-{epoch}.pth')

    print('Training complete!')
    
    return model 