import pickle

from os.path import join


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
    for i, token_ind in enumerate(play_tokens): 
        if token_ind == stop_ind:
            continue
        j = 0
        while j < block_size and i + j + 1 < len(play_tokens): 
            j += 1
            training_data = (play_tokens[i:i + j], play_tokens[i + j])
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