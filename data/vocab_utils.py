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
