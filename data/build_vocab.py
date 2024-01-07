# execute in the data/ directory

import pickle

from os import listdir
from os.path import isfile, join

from collections import defaultdict

from matplotlib import pyplot as plt
from utils import read_corpus, get_char_type

path_to_plays = '../shakespeare/shakespeare-db/'

def build_vocab():
    """Build a dictionary for the input data to map words to indices."""
    vocab = defaultdict(int) 
    plays = [join(path_to_plays, f) for f in listdir(path_to_plays) if isfile(join(path_to_plays, f))]

    for p in plays:
        string = read_corpus(p)
        play_length = len(string)

        vocab['<start>'] += 1

        i = 0
        while i < play_length:
            token = ''
            c_type = get_char_type(string[i])
            j = i
            while j < play_length and get_char_type(string[j]) == c_type:
                j += 1
            token = string[i:j]
            vocab[token] += 1

            # if i < 400:
            #     print(token)

            i = j

        vocab['<stop>'] += 1
    
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
    vocab = build_vocab()
    ind_to_vocab = sorted(vocab.keys(), key=lambda x: vocab[x], reverse=True)
    vocab_to_ind = {v: i for i, v in enumerate(ind_to_vocab)}

    generate_frequency_histgram(vocab, ind_to_vocab)

    with open('../data/vocab_to_ind.pkl', 'wb') as outfile:
        pickle.dump(vocab_to_ind, outfile)
    
    with open('../data/ind_to_vocab.pkl', 'wb') as outfile:
        pickle.dump(ind_to_vocab, outfile)