import pickle
import torch

import numpy as np

from os import listdir
from os.path import isfile, join

from .dataloader import BabyShakespeareDataset

def read_corpus(file_path):
    """Read file, return a list of list of words."""
    play_in_string = ''
    with open(file_path, 'r', encoding='utf-8') as infile:
        play_in_string = infile.read()
    
    # for i in range(10):
    #     for j in range(30):
    #         print(my_str[i*30+j], end='')
    #     input()
    play_in_string = play_in_string.strip()

    return play_in_string


def get_char_type(c):
    """Get the type of the character."""
    if c.isalpha():
        return 'alpha'
    elif c.isdigit():
        return 'digit'
    elif c == ' ':
        return 'space'
    elif c == '\n':
        return 'newline'
    elif c.isspace():
        return 'otherspaces'
    else:
        return 'other'


def tokenize_play(play_string, vocab_to_ind):
    """Tokenize the play string."""

    play_length = len(play_string)
    i = 0
    tokens = [vocab_to_ind['<start>']]
    while i < play_length:
        token = ''
        c_type = get_char_type(play_string[i])
        j = i
        while j < play_length and get_char_type(play_string[j]) == c_type:
            j += 1
        token = play_string[i:j]
        tokens.append(vocab_to_ind[token])
        i = j
    tokens.append(vocab_to_ind['<stop>'])

    return tokens


def generate_dataset_from_tokens(play_tokens, vocab_to_ind, block_size):
    """Generate a sequence of tokens from the play string."""
    stop_ind = vocab_to_ind['<stop>']

    data = []
    for i in range(len(play_tokens) - block_size - 1): 
        # training_data = (play_tokens[i:i + block_size], play_tokens[i + 1: i + block_size + 1])
        data.append((play_tokens[i:i + block_size], play_tokens[i + 1: i + block_size + 1]))
        # data.append(training_data)
        
        # if i % 10000 == 0:
        #     print(f"Generated data at location: {i}, total number of tokens: {len(play_tokens)}")

    return data 


def load_pickled_data(file_name, picked_data_path='./data/'):
    with open(join(picked_data_path, file_name), 'rb') as infile:
        vocab_to_ind = pickle.load(infile)
    
    return vocab_to_ind


def pickle_data(data, file_name, picked_data_path='./data/'):
    """Pickle the data."""
    with open(join(picked_data_path, file_name), 'wb') as outfile:
        pickle.dump(data, outfile)


def load_all_data(vocab_to_ind, factor, block_size=8, shakespeare_path='./shakespeare/shakespeare-db/', data_path='./data/data.npz'):
    plays = [join(shakespeare_path, f) for f in listdir(shakespeare_path) if isfile(join(shakespeare_path, f))]
    block_size = block_size
    data = []
    if not isfile(data_path):
        data = []
        for p in plays:
            print("Play: ", p)
            print("  Reading...")
            play_in_string = read_corpus(p)
            print("  Tokenizing...")
            play_tokens = tokenize_play(play_in_string, vocab_to_ind)
            print("  Generating dataset from tokens...")
            dataset_from_one_play = generate_dataset_from_tokens(play_tokens, vocab_to_ind, block_size)
            print("  Dataset length: ", len(dataset_from_one_play))
            data += dataset_from_one_play
        
        np.savez_compressed('./data/data.npz', data, allow_pickle=False)
    else:
        data = np.load('./data/data.npz', allow_pickle=True)['arr_0']
    
    print("Length of data: ", len(data))
    end_of_selected_data = int(len(data) * factor)
    print("Shape of np data: ", data.shape)
    print("Tensorizing data...")
    data = torch.from_numpy(data)[0:end_of_selected_data].long()
    print("data shape: ", data.shape)
    
    return data


def get_train_and_test_dataset(vocab_to_ind, factor, device='cpu', block_size=8, shakespeare_path='./shakespeare/shakespeare-db/'):
    """Get the training and testing dataset."""
    print("Loading data...")
    data = load_all_data(vocab_to_ind, factor, block_size, shakespeare_path)

    dataset = BabyShakespeareDataset(data, device)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3], torch.Generator(device))
    
    return train_dataset, test_dataset

def generate_contents(model, vocab_to_ind, ind_to_vocab, device='cpu', max_num_of_tokens=1000):
    """Generate contents from the model."""

    output = None
    token_indx = [vocab_to_ind['<start>']]
    with torch.no_grad():
        for i in range(max_num_of_tokens):
            input = torch.tensor(token_indx).unsqueeze(0).to(device)
            if output is None:
                output = model(input, input)
            else:
                output = model(input, torch.tensor(token_indx).unsqueeze(0).to(device))
            output = output[:, -1, :]
            output = torch.softmax(output, dim=-1) #[1, vocab_size]
            output = torch.multinomial(output, num_samples=1)
            token_indx.append(output.item())
            if output.item() == vocab_to_ind['<stop>']:
                break
            print(ind_to_vocab[output.item()], end='')
