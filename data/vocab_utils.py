import os

from collections import defaultdict

from matplotlib import pyplot as plt
from .utils import read_corpus, get_char_type, get_play_paths


def generate_frequency_histgram(vocab, ind_to_vocab, num_of_words=30):
    """Generate a histgram for the frequency of words."""
    words_to_show = ind_to_vocab[0:num_of_words]
    y = [vocab[w] for w in words_to_show]

    def transform_spaces(s):
        """Transform spaces to a readable format."""
        if s[0] == "\n":
            return str(s.count("\n")) + " nls"
        elif s[0].isspace():
            return f"{str(s.count(' '))} sps"
        else:
            return s

    words_to_show = [transform_spaces(w) for w in words_to_show]

    plt.bar(words_to_show, y)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Word frequency of Shakespeare's play")
    plt.xticks(rotation=90)
    plt.savefig("../figs/word_frequency.png", dpi=600)


def build_char_vocab(play_paths):
    vocab = defaultdict(int)

    for p in play_paths:
        string = read_corpus(p)

        for i, c in enumerate(string):
            vocab[c] += 1
    return vocab


def build_word_vocab(play_paths):
    """Build a dictionary for the input data to map words to indices."""
    vocab = defaultdict(int)

    for p in play_paths:
        string = read_corpus(p)
        play_length = len(string)

        vocab["<start>"] += 1

        i = 0
        while i < play_length:
            token = ""
            c_type = get_char_type(string[i])
            j = i
            while j < play_length and get_char_type(string[j]) == c_type:
                j += 1
            token = string[i:j]
            vocab[token] += 1

            i = j

        vocab["<stop>"] += 1

    return vocab


def build_vocab(tokenizer, play_paths):
    if tokenizer == "char":
        vocab = build_char_vocab(play_paths)
    elif tokenizer == "word":
        vocab = build_word_vocab(play_paths)
    return vocab


def get_vocab(tokenizer, play_paths):
    vocab = build_vocab(tokenizer, play_paths)
    ordered_vocabs = sorted(vocab.keys(), key=lambda x: vocab[x], reverse=True)
    vocab_to_ind = {v: i for i, v in enumerate(ordered_vocabs)}

    return vocab_to_ind
