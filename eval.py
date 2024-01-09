import argparse

import torch
import torch.nn as nn

from data.utils import load_pickled_data, get_train_and_test_dataset, generate_contents
from models.transformer import Transformer
from hyperparams import *


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--epoch', type=int, default=150)
    parser = argparser.parse_args()
    epoch = parser.epoch
    print("Hello World!")
    vocab_to_ind = load_pickled_data('vocab_to_ind.pkl') 
    ind_to_vocab = load_pickled_data('ind_to_vocab.pkl')
    torch.set_default_device(device)
    print("Token type number: ", len(vocab_to_ind))

    model = Transformer(len(vocab_to_ind), dropout=0.2, block_size=block_size, num_of_decoder_layers=2, num_of_encoder_layers=2, dmodel=dmodel).to(device) 
    model.load_state_dict(torch.load(f'data/model-{epoch}.pth'))
    model.eval()

    generate_contents(model, vocab_to_ind, ind_to_vocab=ind_to_vocab, device=device, max_num_of_tokens=1000)
    