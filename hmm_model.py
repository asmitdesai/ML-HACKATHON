import numpy as np
import pickle
import config

class HangmanHMM:
    """
    Implements the Hidden Markov Model to act as a probabilistic "oracle"[cite: 18].
    """
    def __init__(self, n_states=config.HMM_N_STATES):
        self.n_states = n_states
        self.n_emissions = len(config.ALPHABET)
        self.letter_to_idx = {letter: i for i, letter in enumerate(config.ALPHABET)}
        
        # HMM parameters (to be learned)
        # You must decide how to handle different word lengths [cite: 49]
        self.transition_probs = None
        self.emission_probs = None
        self.initial_probs = None
        print(f"HMM initialized with {self.n_states} hidden states.")

    def train(self, words):
        """
        Trains the HMM parameters on the provided corpus[cite: 17].
        This is where you'll implement the Baum-Welch (forward-backward)
        algorithm.
        """
        print("Training HMM... (This is the part you need to build!)")
        # ---
        # Your HMM training logic (e.g., Baum-Welch) goes here.
        # ---
        print("HMM Training Complete.")
        
    def get_letter_probabilities(self, masked_word, guessed_letters):
        """
        Calculates the probability distribution over the alphabet
        for the *next* guess[cite: 19, 23].
        
        This is the "oracle" output for the RL agent.
        """
        # ---
        # Your HMM inference logic (e.g., forward algorithm) goes here.
        # You need to calculate the probability of each *unguessed* letter
        # appearing in any of the '_' spots.
        # ---
        
        # Placeholder: Return a uniform distribution over unguessed letters
        available_letters = [l for l in config.ALPHABET if l not in guessed_letters]
        if not available_letters:
            return {l: 0.0 for l in config.ALPHABET}
            
        prob = 1.0 / len(available_letters)
        
        # Return a dictionary: {'a': 0.05, 'b': 0.0, ...}
        prob_dict = {l: 0.0 for l in config.ALPHABET}
        for letter in available_letters:
            prob_dict[letter] = prob # Replace this with your HMM's calculated prob
            
        return prob_dict

    def save(self, path=config.HMM_MODEL_PATH):
        """Saves the trained HMM model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"HMM model saved to {path}")

    @staticmethod
    def load(path=config.HMM_MODEL_PATH):
        """Loads a pre-trained HMM model from disk."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"HMM model loaded from {path}")
        return model
