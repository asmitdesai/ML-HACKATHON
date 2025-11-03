#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import os
corpus_file =  'corpus.txt'

# Use a defaultdict to automatically handle new list creation
words_by_length = collections.defaultdict(list)

print(f"Starting to process {corpus_file}...")

# Check if the file exists before trying to open it
if not os.path.exists(corpus_file):
    print(f"Error: The file '{corpus_file}' was not found.")
    print("Please make sure your repository structure is correct.")
else:
    with open(corpus_file, 'r') as f:
        for line in f:
            # 1. Strip leading/trailing whitespace and convert to lowercase
            word = line.strip().lower()
            
            # 2. Filter out any lines that aren't purely alphabetic
            if word.isalpha():
                # 3. Group the word by its length
                words_by_length[len(word)].append(word)

    print("Corpus processing complete.")

    print(f"Found words for {len(words_by_length)} different lengths.")
    
    lengths_sorted = sorted(words_by_length.keys())
    
    if lengths_sorted:
        print("\n--- Word Counts Per Length (Sample) ---")
        sample_lengths = lengths_sorted[:5] + lengths_sorted[-5:]
        for length in sample_lengths:
            print(f"Length {length}: {len(words_by_length[length])} words")
            
        if 7 in words_by_length:
            print("\nExample (first 5 words of length 7):")
            print(words_by_length[7][:5])


# In[2]:


import string
import pickle
import collections
from collections import Counter

# --- Utility: Convert defaultdicts to normal dicts for pickling ---
def to_dict(obj):
    """Recursively convert defaultdicts to normal dicts for pickling."""
    if isinstance(obj, collections.defaultdict):
        obj = {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        obj = {k: to_dict(v) for k, v in obj.items()}
    return obj

# --- HMM Configuration ---
models_output_file = 'hmm_models.pkl'  # Output file for trained models
ALPHABET = list(string.ascii_lowercase)
NUM_CLASSES = len(ALPHABET)

print("\nStarting HMM training...")

# Main model storage.
# Format: { length: {'pi': {...}, 'A': {...}} }
hmm_models = {}

for length, word_list in words_by_length.items():
    if not word_list or length < 2:
        continue 

    total_words = len(word_list)

    # --- Calculate Pi (Initial Probabilities) ---
    pi_counts = Counter(word[0] for word in word_list)
    pi_probs = {char: (pi_counts[char] + 1) / (total_words + NUM_CLASSES)
                for char in ALPHABET}

    # --- Calculate A (Transition Probabilities) ---
    A_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    A_totals = collections.defaultdict(int)

    for word in word_list:
        for i in range(length - 1):
            prev_char = word[i]
            next_char = word[i + 1]
            A_counts[prev_char][next_char] += 1
            A_totals[prev_char] += 1

    A_probs = collections.defaultdict(lambda: collections.defaultdict(float))
    for prev_char in ALPHABET:
        total_prev_transitions = A_totals[prev_char]
        for next_char in ALPHABET:
            count = A_counts[prev_char][next_char]
            A_probs[prev_char][next_char] = (count + 1) / (total_prev_transitions + NUM_CLASSES)

    hmm_models[length] = {
        'pi': pi_probs,
        'A': A_probs
    }

print("\nHMM training complete.")
print(f"Trained {len(hmm_models)} models (one for each word length).")

# --- Convert all defaultdicts to dicts before saving ---
hmm_models_clean = to_dict(hmm_models)

# --- Save Models ---
with open(models_output_file, 'wb') as f:
    pickle.dump(hmm_models_clean, f)

print(f"Models saved to '{models_output_file}' successfully.")


# In[3]:


# --- Phase 3: Save models to disk for later use ---
import collections

def to_dict(obj):
    """Recursively convert defaultdicts to normal dicts for pickling."""
    if isinstance(obj, collections.defaultdict):
        obj = {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        obj = {k: to_dict(v) for k, v in obj.items()}
    return obj

# Convert defaultdicts before saving
hmm_models_clean = to_dict(hmm_models)

# Use pickle to serialize the main 'hmm_models' dict
try:
    with open(models_output_file, 'wb') as f:
        pickle.dump(hmm_models_clean, f)
    print(f"Successfully saved trained models to '{models_output_file}'")
except Exception as e:
    print(f"Error saving models: {e}")


# In[4]:


# --- Phase 4: Verification (Load models & check) ---
import pickle

try:
    with open(models_output_file, 'rb') as f:
        loaded_models = pickle.load(f)

    print(f"\n--- Verification: Loaded {len(loaded_models)} models ---")

    # Pick a common word length (e.g., 7)
    test_length = 7
    if test_length in loaded_models:
        model = loaded_models[test_length]
        print(f"\nChecking model for word length = {test_length}...")

        # --- Initial Probabilities ---
        print("\nInitial Probabilities (Pi) sample:")
        print(f"P('s'...) = {model['pi'].get('s', 0):.6f}")  # likely higher
        print(f"P('z'...) = {model['pi'].get('z', 0):.6f}")  # likely lower

        # --- Transition Probabilities ---
        print("\nTransition Probabilities (A) sample:")
        q_row = model['A'].get('q', {})
        if q_row:
            print(f"P('u' | 'q') = {q_row.get('u', 0):.6f}")
            print(f"P('x' | 'q') = {q_row.get('x', 0):.6f}")
        else:
            print("No transitions found for 'q' (possibly no words with 'q').")

        # --- Sanity check: normalization ---
        for ch in ['a', 'q', 's']:
            total_prob = sum(model['A'][ch].values())
            print(f"Sum of P(* | '{ch}') = {total_prob:.6f}")
            if abs(total_prob - 1.0) > 1e-3:
                print(f"Warning: probabilities for '{ch}' not normalized!")

    else:
        print(f"No model found for word length = {test_length}.")

except FileNotFoundError:
    print(f"Verification failed: '{models_output_file}' not found.")
except Exception as e:
    print(f"Verification failed: {e}")


# In[5]:


import pickle
import string
import os

# --- Constants ---
ALPHABET = list(string.ascii_lowercase)
ALPHABET_SET = set(ALPHABET)
MODELS_FILE = 'hmm_models.pkl'

# --- Load Models ---
hmm_models = None # Will hold all trained HMMs (pi, A)

try:
    with open(MODELS_FILE, 'rb') as f:
        hmm_models = pickle.load(f)
    print(f"Successfully loaded HMM models from '{MODELS_FILE}'.")
    print(f"Loaded {len(hmm_models)} models for lengths: {sorted(hmm_models.keys())}")
except FileNotFoundError:
    print(f"ERROR: Model file '{MODELS_FILE}' not found.")
    print("Please run the training cell (Cell 2) first.")
except Exception as e:
    print(f"Error loading models: {e}")


# In[6]:


import collections

def get_letter_probs(masked_word, guessed_letters, smoothing=1e-8):
    """
    Computes posterior probabilities for each unguessed letter 
    given the current masked word state using a trained HMM model.
    
    Args:
        masked_word (str): Current state of the word (e.g., "_ppl_").
        guessed_letters (set): Letters already guessed (e.g., {'p', 'l', 'e'}).
        smoothing (float): Small constant to avoid zero probabilities.

    Returns:
        dict: Normalized probabilities {letter: probability} for each unguessed letter.
    """
    # --- Setup ---
    L = len(masked_word)
    if not hmm_models or L not in hmm_models:
        print(f"[WARN] No HMM model found for length {L}. Using uniform distribution.")
        unguessed = [c for c in ALPHABET if c not in guessed_letters]
        return {c: 1.0 / len(unguessed) for c in unguessed} if unguessed else {}

    model = hmm_models[L]
    pi, A = model['pi'], model['A']

    # --- Step 1: Build Emission Probabilities (B matrix) ---
    B = []
    for obs in masked_word:
        emission = {}
        for char in ALPHABET:
            if obs == '_':
                emission[char] = 1.0 if char not in guessed_letters else 0.0
            else:
                emission[char] = 1.0 if char == obs else 0.0
        B.append(emission)

    # --- Step 2: Forward Pass ---
    alpha = [{} for _ in range(L)]
    for char in ALPHABET:
        alpha[0][char] = pi[char] * B[0][char] + smoothing

    for t in range(1, L):
        for char_next in ALPHABET:
            alpha[t][char_next] = sum(
                alpha[t-1][char_prev] * A[char_prev][char_next]
                for char_prev in ALPHABET
            ) * B[t][char_next] + smoothing

    prob_obs = sum(alpha[L-1].values())
    if prob_obs <= 0:
        print("[WARN] Observation probability is zero. Falling back to uniform.")
        unguessed = [c for c in ALPHABET if c not in guessed_letters]
        return {c: 1.0 / len(unguessed) for c in unguessed} if unguessed else {}

    # --- Step 3: Backward Pass ---
    beta = [{char: 1.0 for char in ALPHABET} for _ in range(L)]
    for t in range(L - 2, -1, -1):
        for char_prev in ALPHABET:
            beta[t][char_prev] = sum(
                A[char_prev][char_next] * B[t + 1][char_next] * beta[t + 1][char_next]
                for char_next in ALPHABET
            ) + smoothing

    # --- Step 4: Compute Posterior Probabilities ---
    posterior = collections.defaultdict(float)
    for t, obs in enumerate(masked_word):
        if obs == '_':
            for char in ALPHABET:
                if char not in guessed_letters:
                    posterior[char] += (alpha[t][char] * beta[t][char]) / prob_obs

    total = sum(posterior.values())
    if total <= 0:
        return {c: 0.0 for c in ALPHABET if c not in guessed_letters}

    return {char: prob / total for char, prob in posterior.items()}


# In[7]:


# --- Test the 'get_letter_probs' function ---

print("\n========== HMM Oracle Verification ==========\n")

def run_test(masked, guessed, expected_top=None):
    """Helper function for testing get_letter_probs."""
    print(f"Masked word: {masked}")
    print(f"Guessed letters: {sorted(list(guessed))}")
    probs = get_letter_probs(masked, guessed)

    if not probs:
        print("No probabilities returned.")
        print("-" * 50)
        return

    # Check that probabilities sum to ~1
    total_prob = sum(probs.values())
    print(f"Sum of probabilities: {total_prob:.6f}")

    # Sort by descending probability
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_probs[:3]

    print(f"Top 3 predicted letters: {top_3}")

    # Optionally highlight expected letter
    if expected_top:
        rank = next((i+1 for i, (c, _) in enumerate(sorted_probs) if c == expected_top), None)
        if rank:
            print(f"'{expected_top}' found at rank {rank} with P={probs[expected_top]:.4f}")
        else:
            print(f"Expected letter '{expected_top}' not found among probable guesses.")

    print("-" * 50)


# --- Run Tests ---
print("Running HMM tests...\n")

run_test(masked="appl_", guessed={'a', 'p', 'l'}, expected_top='e')
run_test(masked="pr_ject", guessed={'p', 'r', 'j', 'e', 'c', 't'}, expected_top='o')

print("\n========== Tests Complete ==========\n")


# In[ ]:




