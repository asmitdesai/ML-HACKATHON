import config

def load_corpus(path=config.CORPUS_PATH):
    """
    Loads and filters the corpus from a text file[cite: 36].
    """
    with open(path, 'r') as f:
        words = [
            word.strip().lower() for word in f
            if word.strip() and all(c in config.ALPHABET for c in word.strip())
        ]
    print(f"Loaded {len(words)} valid words from {path}.")
    return words

def mask_word(word, guessed_letters):
    """
    Creates the masked word state (e.g., '_ p p l _')[cite: 30].
    """
    return ''.join([c if c in guessed_letters else '_' for c in word])

def word_to_one_hot(masked_word, alphabet=config.ALPHABET):
    """
    Converts a masked word (e.g., '_a_') into a 26-dim one-hot vector.
    Note: This is a simple state representation; a more complex one
    might be needed[cite: 52].
    """
    vec = [0.0] * len(alphabet)
    for char in masked_word:
        if char != '_':
            vec[alphabet.index(char)] = 1.0
    return vec

def guessed_to_binary_vec(guessed_letters, alphabet=config.ALPHABET):
    """
    Converts the set of guessed letters into a 26-dim binary vector[cite: 52].
    """
    return [1.0 if char in guessed_letters else 0.0 for char in alphabet]
