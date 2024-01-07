import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F

from tqdm import tqdm
from matplotlib import pyplot as plt

from models.transformer import Transformer 
from data.dataloader import BabyShakespeareDataset
from data.utils import load_pickled_data


def train(model, train_loader, criterion, optimizer, num_classes, device, epochs=1):
    # Define loss function and optimizer
    # Training loop
    loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients
                logits = model(inputs, labels)  # Forward pass: (B, T, Emb)
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                labels = labels.view(B * T)

                loss = criterion(logits, labels)  # Compute the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                running_loss += loss.item()
                pbar.update(1)  # Update the progress bar
        print(f'Loss: {running_loss / len(train_loader)}')
        plt.pause(0.001)
        # input("Press Enter to continue...")

        loss_history.append(running_loss/ len(train_loader))    
    
        plt.plot(range(epoch + 1), loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History')

    plt.show()
    
    print('Training complete!')
    return model 


if __name__ == "__main__":
    print("Hello World!")
    print("CUDA available: ", torch.cuda.is_available())
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    block_size = 8
    batch_size = 8 
    dmodel = 256 

    learning_rate = 0.0001

    vocab_to_ind = load_pickled_data('vocab_to_ind.pkl') 

    train_dataset = BabyShakespeareDataset(vocab_to_ind, block_size=block_size, dataset_size=2045795//10, device=device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

    model = Transformer(len(vocab_to_ind), block_size=block_size, dmodel=dmodel).to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create an instance of your model
    model = train(model, train_loader, criterion, optimizer, num_classes=len(vocab_to_ind), epochs=10, device=device)

    # Save the model
    torch.save(model.state_dict(), './data/model.pth')