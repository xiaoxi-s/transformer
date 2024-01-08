import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F

from tqdm import tqdm
from matplotlib import pyplot as plt

from models.transformer import Transformer 
from data.utils import load_pickled_data, get_train_and_test_dataset


def train(model, train_loader, test_loader, criterion, optimizer, epochs=1):
    # Define loss function and optimizer
    # Training loop
    train_loss_history = []
    test_loss_history = []
    epoch_sequence = []

    plt.ion()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')

    for epoch in range(epochs):
        # training
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            model.train()
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
        train_loss_history.append(running_loss/ len(train_loader))    

        # testing
        running_loss = 0.0    
        with torch.no_grad():
            model.eval()
            for inputs, labels in test_loader:
                logits = model(inputs, labels)  # Forward pass: (B, T, Emb)
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                labels = labels.view(B * T)

                loss = criterion(logits, labels)  # Compute the loss
                running_loss += loss.item()
        test_loss_history.append(running_loss/ len(test_loader))

        epoch_sequence.append(epoch + 1)

        plt.plot(epoch_sequence, train_loss_history, 'b-', label='train loss')
        plt.plot(epoch_sequence, test_loss_history, 'r-', label='test loss')
        plt.show(block=False)

        if epoch == 0:
            plt.legend()

        plt.pause(0.001)

    plt.ioff()
    plt.show()

    print('Training complete!')
    
    return model 


if __name__ == "__main__":
    print("Hello World!")
    print("CUDA available: ", torch.cuda.is_available())
    torch.manual_seed(7777)
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    block_size = 8
    batch_size = 32 
    dmodel = 128 
    learning_rate = 0.00001
    vocab_to_ind = load_pickled_data('vocab_to_ind.pkl') 

    length_of_data = 2045795
    total_length_of_data_for_model = int(length_of_data * 0.01)
    train_data_length = int(total_length_of_data_for_model * 0.9)
    test_data_length = total_length_of_data_for_model - train_data_length

    train_dataset, test_dataset = get_train_and_test_dataset(vocab_to_ind, 0, train_data_length, train_data_length, train_data_length + test_data_length, device=device, block_size=block_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

    model = Transformer(len(vocab_to_ind), block_size=block_size, num_of_decoder_layers=2, num_of_encoder_layers=2, dmodel=dmodel).to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create an instance of your model
    model = train(model, train_loader, test_loader, criterion, optimizer, epochs=10)

    # Save the model
    torch.save(model.state_dict(), './data/model.pth')