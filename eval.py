import argparse

import torch
import torch.nn as nn

import numpy as np

from data.utils import load_pickled_data, get_train_and_test_dataset, generate_contents
from models.transformer import Transformer
from hyperparams import *


if __name__ == "__main__":
    np.random.seed(7777)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-name', '--model-name', type=str)
    argparser.add_argument('-m', '--max-token', type=int, default=1000)
    argparser.add_argument('-p', '--parallel', default="true", type=str)      # option that takes a value
    argparser.add_argument('-t', '--tokenizer', default='char', type=str)           # positional argument

    parser = argparser.parse_args()
    model_name = parser.model_name
    max_token = parser.max_token
    tokenizer = parser.tokenizer

    print("Loading vocab ...")
    if tokenizer == 'char':
        vocab_to_ind = load_pickled_data('char_vocab_to_ind.pkl')
        ind_to_vocab = load_pickled_data('char_ind_to_vocab.pkl')
    elif tokenizer == 'word':
        vocab_to_ind = load_pickled_data('word_vocab_to_ind.pkl')
        ind_to_vocab = load_pickled_data('word_ind_to_vocab.pkl')
    print("Done loading vocab ...")
    torch.set_default_device(device)
    print("Vocab size: ", len(vocab_to_ind))

    model = Transformer(len(vocab_to_ind), dropout=dropout, block_size=block_size, num_of_decoder_layers=1, num_of_encoder_layers=1, dmodel=dmodel).to(device) 
    if parser.parallel.lower() == "true" or parser.parallel.lower() == "t":
        model = nn.DataParallel(Transformer(len(vocab_to_ind), dropout=dropout, block_size=block_size, num_of_decoder_layers=1, num_of_encoder_layers=1, dmodel=dmodel).to(device)) 

    model.load_state_dict(torch.load(f'data/{model_name}'))
    model.eval()

    generate_contents(model, vocab_to_ind, ind_to_vocab=ind_to_vocab, device=device, max_num_of_tokens=max_token)
    