import torch
import torch.nn as nn
import torch.optim as optim

from models.transformer import Transformer 
from data.dataloader import BabyShakespeareDataset
from data.utils import load_pickled_data
from tqdm import tqdm

from matplotlib import pyplot as plt


def train(model, train_loader, criterion, optimizer, num_classes, epochs=1):
    # Define loss function and optimizer
    # Training loop
    loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients
                outputs = model(inputs, labels)  # Forward pass
                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(torch.float32) 
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                running_loss += loss.item()
                pbar.update(1)  # Update the progress bar

        loss_history.append(running_loss)    
    
        plt.plot(range(epoch + 1), loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History')

        plt.pause(0.001)
    plt.show()
    
    print('Training complete!')
    return loss_history


if __name__ == "__main__":
    print("Hello World!")
    print("CUDA available: ", torch.cuda.is_available())
    cuda_available = torch.cuda.is_available()

    block_size = 8
    batch_size = 4 
    dmodel = 256 

    learning_rate = 0.001

    vocab_to_ind = load_pickled_data('vocab_to_ind.pkl') 

    train_dataset = BabyShakespeareDataset(vocab_to_ind, block_size=block_size, dataset_size=20)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Transformer(len(vocab_to_ind), block_size=block_size, dmodel=dmodel) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create an instance of your model
    train(model, train_loader, criterion, optimizer, num_classes=len(vocab_to_ind), epochs=3)