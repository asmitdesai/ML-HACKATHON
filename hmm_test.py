import pickle
import string

# --- 1. LOAD TRAINED MODELS ---
# This code runs ONCE when the script is imported.
# It loads the model file Asmit's notebook created.

MODELS_FILE = 'hmm_models.pkl'
ALPHABET = list(string.ascii_lowercase)
ALL_MODELS = {}

try:
    with open(MODELS_FILE, 'rb') as f:
        ALL_MODELS = pickle.load(f)
    print(f"--- HMM: Successfully loaded {len(ALL_MODELS)} models from {MODELS_FILE} ---")
except FileNotFoundError:
    print(f"--- HMM ERROR: '{MODELS_FILE}' not found. ---")
    print("--- HMM: Please run the HMM.ipynb notebook first! ---")
    ALL_MODELS = None
except Exception as e:
    print(f"--- HMM ERROR: Could not load models. {e} ---")
    ALL_MODELS = None


# --- 2. THE PREDICTION FUNCTION ---
# This is the function your RL agent will call.

def get_letter_probabilities(masked_word, guessed_letters):
    """
    Estimates the probability of each letter appearing in the blanks,
    based on the trained HMMs.
    """
    
    # Use lowercase for all logic
    masked_word = masked_word.lower()
    guessed_letters = [l.lower() for l in guessed_letters]
    
    word_len = len(masked_word)

    # --- Safety Checks ---
    if not ALL_MODELS or word_len not in ALL_MODELS:
        # Fallback: If no model, return uniform probability
        prob = 1.0 / (26 - len(guessed_letters)) if (26 - len(guessed_letters)) > 0 else 0
        return {char: prob for char in ALPHABET if char not in guessed_letters}

    # --- Get the correct model for this word length ---
    model = ALL_MODELS[word_len]
    pi_probs = model.get('pi', {})
    A_probs = model.get('A', {}) # Transition probabilities

    # This will hold the total "score" or "likelihood" for each letter
    letter_scores = {char: 0.0 for char in ALPHABET}
    
    # Iterate over each position in the word
    for i, char in enumerate(masked_word):
        if char == '_': # This is a blank we need to fill
            
            # --- Case 1: First letter is blank ---
            if i == 0:
                for letter in ALPHABET:
                    letter_scores[letter] += pi_probs.get(letter, 0)
            
            # --- Case 2: Not the first letter ---
            else:
                prev_char = masked_word[i-1]
                if prev_char != '_':
                    # The previous letter is known! Use the transition probability.
                    # P(current_letter | prev_char)
                    transitions = A_probs.get(prev_char, {})
                    for letter in ALPHABET:
                        letter_scores[letter] += transitions.get(letter, 0)
                else:
                    # The previous letter is also blank.
                    # This simple model can't look back further.
                    # We'll just add the overall initial probability (pi)
                    # as a general-purpose "best guess".
                    for letter in ALPHABET:
                        letter_scores[letter] += pi_probs.get(letter, 0)

    # --- 3. FINALIZE & NORMALIZE ---
    
    final_probs = {}
    
    # Filter out letters that have already been guessed
    for letter in ALPHABET:
        if letter not in guessed_letters:
            final_probs[letter] = letter_scores.get(letter, 0)
        else:
            final_probs[letter] = 0.0 # Zero probability if already guessed

    # Normalize the scores to be a valid probability distribution
    total_score = sum(final_probs.values())
    
    if total_score > 0:
        for letter in final_probs:
            final_probs[letter] = final_probs[letter] / total_score
    else:
        # Failsafe if all scores are 0 (e.g., all letters guessed)
        # Just return a uniform distribution over remaining
        remaining = [l for l in ALPHABET if l not in guessed_letters]
        if remaining:
            prob = 1.0 / len(remaining)
            for letter in remaining:
                final_probs[letter] = prob
                
    return final_probs