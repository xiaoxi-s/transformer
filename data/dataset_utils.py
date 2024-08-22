import pickle
import torch

import numpy as np

from os import listdir
from os.path import isfile, join

from .dataloader import BabyShakespeareDataset
from .utils import read_corpus, get_char_type


def word_tokenize_play(play_string, vocab_to_ind):
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

def char_tokenize_play(play_string, vocab_to_ind):
    chars = list(play_string)
    tokens = list(map(lambda c: vocab_to_ind[c], chars))
    return tokens


def generate_dataset_from_tokens(play_tokens, block_size):
    """Generate a sequence of tokens from the play string."""

    data = []
    for i in range(0, len(play_tokens) - block_size - 1, block_size): 
        if i + block_size + 1 < len(play_tokens):
            data.append((play_tokens[i:i + block_size], play_tokens[i + 1: i + block_size + 1]))
        elif i + block_size + 1 >= len(play_tokens): 
            if i == len(play_tokens) - 1:
                break
            else:
                data.append((play_tokens[i: -1], play_tokens[i + 1:]))

    return data 


def load_dataset(vocab_to_ind, tokenizer, play_paths, block_size=8):
    if tokenizer == 'char':
        tokenizer_func = char_tokenize_play
    else:
        tokenizer_func = word_tokenize_play

    block_size = block_size
    data = []

    for p in play_paths:
        print("Play: ", p)
        print("  Reading...")
        play_in_string = read_corpus(p)
        print("  Tokenizing...")
        play_tokens = tokenizer_func(play_in_string, vocab_to_ind)
        print("  Generating dataset from tokens...")
        dataset_from_one_play = generate_dataset_from_tokens(play_tokens, block_size)
        print("  Dataset length: ", len(dataset_from_one_play))
        data += dataset_from_one_play

    print("Length of data: ", len(data))
    print("Tensorizing data...")
    data = torch.Tensor(data).long()
    print("data shape: ", data.shape)

    return data


def generate_dataset(vocab_to_ind, play_paths, tokenizer, block_size=8):
    """Get the training and testing dataset."""
    data = load_dataset(vocab_to_ind, tokenizer, play_paths, block_size)
    dataset = BabyShakespeareDataset(data)
    return dataset

