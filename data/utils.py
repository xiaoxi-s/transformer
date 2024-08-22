import os

def read_corpus(file_path):
    """Read file, return a list of list of words."""
    play_in_string = ""
    with open(file_path, "r", encoding="utf-8") as infile:
        play_in_string = infile.read()
    play_in_string = play_in_string.strip()
    return play_in_string


def get_char_type(c):
    """Get the type of the character."""
    if c.isalpha():
        return "alpha"
    elif c.isdigit():
        return "digit"
    elif c == " ":
        return "space"
    elif c == "\n":
        return "newline"
    elif c.isspace():
        return "otherspaces"
    else:
        return "other"


def get_play_paths(dataset_path):
    # accomadate for single txt file (the preprocessed dataset) vs. a directory of txt files (the raw dataset)
    if os.path.isdir(dataset_path):
        play_paths = [
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if os.path.isfile(os.path.join(dataset_path, f))
        ]
    elif os.path.isfile(dataset_path):
        play_paths = [dataset_path]

    return play_paths