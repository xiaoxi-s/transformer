import pickle

from os import listdir
from os.path import isfile, join

from collections import defaultdict

from matplotlib import pyplot as plt
from vocab_utils import read_corpus, get_char_type

path_to_plays = '../shakespeare/shakespeare-db/'

def build_vocab():
    vocab = defaultdict(int) 
    plays = [join(path_to_plays, f) for f in listdir(path_to_plays) if isfile(join(path_to_plays, f))]

    for p in plays:
        string = read_corpus(p)

        for i, c in enumerate(string):
            vocab[c] += 1
    return vocab

if __name__ == '__main__':
    vocab = build_vocab()
    ind_to_vocab = sorted(vocab.keys(), key=lambda x: vocab[x], reverse=True)
    vocab_to_ind = {v: i for i, v in enumerate(ind_to_vocab)}

    with open('../data/char_vocab_to_ind.pkl', 'wb') as outfile:
        pickle.dump(vocab_to_ind, outfile)
    
    with open('../data/ind_to_vocab_char.pkl', 'wb') as outfile:
        pickle.dump(ind_to_vocab, outfile)