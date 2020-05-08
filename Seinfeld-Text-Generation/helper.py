from collections import Counter
import os, pickle


SPECIAL_WORDS = {'PADDING': '<PAD>'}


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)
    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))
    # Standardize by converting everyting to lower case
    text = text.lower()
    text = text.split()
    # tokenize
    vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
    int_text = [vocab_to_int[word] for word in text]
    # save the ddicts
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    words_frequency = Counter(text)
    sorted_frequency = sorted(words_frequency, key=words_frequency.get, reverse=True)
    int_to_vocab = {ii: ch for ii, ch in enumerate(sorted_frequency)}
    vocab_to_int = {ch: ii for ii, ch in int_to_vocab.items()}
    # return tuple
    return (vocab_to_int, int_to_vocab)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    token_dict = {'.': "||Period||",
                  ',': "||Comma||",
                  '"': "||Quotationmark||",
                  ';': "||Semicolon||",
                  '!': "||Exclamationmark||",
                  '?': "||Questionmark||",
                  '(': "||LeftParentheses||",
                  ')': "||RightParentheses||",
                  '-': "||Dash||",
                  '\n': "||Return||"}
    return token_dict
