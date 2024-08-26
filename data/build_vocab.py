# execute in the data/ directory

import argparse
import pickle

from os import listdir
from os.path import isfile, join

from collections import defaultdict

from matplotlib import pyplot as plt
from vocab_utils import read_corpus, get_char_type


def build_char_vocab(plays):
    vocab = defaultdict(int)

    for p in plays:
        string = read_corpus(p)

        for _, c in enumerate(string):
            vocab[c] += 1
    return vocab


def build_word_vocab(plays):
    """Build a dictionary for the input data to map words to indices."""
    vocab = defaultdict(int) 

    for p in plays:
        string = read_corpus(p)
        play_length = len(string)

        i = 0
        while i < play_length:
            token = ''
            c_type = get_char_type(string[i])
            j = i
            while j < play_length and get_char_type(string[j]) == c_type:
                j += 1
            token = string[i:j]
            vocab[token] += 1

            i = j

    return vocab


def generate_frequency_histgram(vocab, ind_to_vocab, num_of_words=30):
    """Generate a histgram for the frequency of words."""
    words_to_show = ind_to_vocab[0:num_of_words]
    y = [vocab[w] for w in words_to_show]

    def transform_spaces(s):
        """Transform spaces to a readable format."""
        if s[0] == '\n':
            return str(s.count('\n')) + " nls"
        elif s[0].isspace():
            return f"{str(s.count(' '))} sps"
        else:
            return s

    words_to_show = [transform_spaces(w) for w in words_to_show]

    plt.bar(words_to_show, y)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Word frequency of Shakespeare\'s play')
    plt.xticks(rotation=90)
    plt.savefig('../figs/word_frequency.png', dpi=600)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='shakespear-tokenizer',
                    description='tokenizer for shakespeare transformer')
    parser.add_argument('-t', '--tokenizer', default='char', type=str)           # positional argument
    parser.add_argument('-d', '--dataset-name', default='default', type=str)           # positional argument

    args = parser.parse_args()
    tokenizer = args.tokenizer
    dataset_name = args.dataset_name

    vocab_name = f"{tokenizer}_vocab_to_ind_on_{dataset_name}.pkl"
    vocab_to_ind_path = f'../data/{vocab_name}'
    if dataset_name == 'default':
        dataset_path = '../shakespeare/shakespeare-db/'
    elif dataset_name == 'preprocessed':
        dataset_path = '../input.txt'
    else:
        raise ValueError('Dataset not recognized')
    if isfile(dataset_path):
        plays = [dataset_path]
    else:
        plays = [join(dataset_path, f) for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

    if tokenizer == 'char':
        vocab = build_char_vocab(plays)
    elif tokenizer == 'word':
        vocab = build_word_vocab(plays)
    else:
        raise ValueError('Tokenizer not recognized')

    ind_to_vocab = sorted(vocab.keys(), key=lambda x: vocab[x], reverse=True)
    vocab_to_ind = {v: i for i, v in enumerate(ind_to_vocab)}

    with open(vocab_to_ind_path, 'wb') as outfile:
        pickle.dump(vocab_to_ind, outfile)