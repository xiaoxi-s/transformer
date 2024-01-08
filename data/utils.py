import pickle
import torch

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
        training_data = (play_tokens[i:i + block_size], play_tokens[i + 1: i + block_size + 1])
        data.append(training_data)
        
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


def load_all_data(vocab_to_ind, block_size=8, shakespeare_path='./shakespeare/shakespeare-db/', data_path='./data/data.pkl'):
    plays = [join(shakespeare_path, f) for f in listdir(shakespeare_path) if isfile(join(shakespeare_path, f))]
    block_size = block_size
    if not isfile(data_path):
        for p in plays:
            print("Play: ", p)
            print("  Reading...")
            play_in_string = read_corpus(p)
            print("  Tokenizing...")
            play_tokens = tokenize_play(play_in_string, vocab_to_ind)
            print("  Generating dataset from tokens...")
            dataset_from_one_play = generate_dataset_from_tokens(play_tokens, vocab_to_ind, block_size)
            print("  Dataset length: ", len(dataset_from_one_play))
            data = data + dataset_from_one_play
        
        pickle_data(data, 'data.pkl')
    else:
        data = load_pickled_data('data.pkl')
    
    return data


def get_train_and_test_dataset(vocab_to_ind, train_dataset_start, train_dataset_end, test_dataset_start, test_dataset_end, device='cpu', block_size=8, shakespeare_path='./shakespeare/shakespeare-db/'):
    """Get the training and testing dataset."""
    data = load_all_data(vocab_to_ind, block_size, shakespeare_path)

    train_data = data[train_dataset_start:train_dataset_end]
    test_data = data[test_dataset_start:test_dataset_end]

    for i in range(len(train_data)):
        train_data[i] = (torch.tensor(train_data[i][0]).to(device), torch.tensor(train_data[i][1]).to(device))

    for i in range(len(test_data)):
        test_data[i] = (torch.tensor(test_data[i][0]).to(device), torch.tensor(test_data[i][1]).to(device))
    
    train_dataset = BabyShakespeareDataset(train_data)
    test_dataset = BabyShakespeareDataset(test_data)
    
    return train_dataset, test_dataset
