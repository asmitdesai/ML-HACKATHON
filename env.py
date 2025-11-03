# FILE: environment.py

import random

class HangmanEnv:
    """
    MOCK ENVIRONMENT: A simple Hangman game that follows the 
    rules and reward structure.
    """
    def __init__(self, corpus_path=None):
        # We ignore the corpus path for this mock
        self.word_list = ["PYTHON", "HACKATHON", "AGENT"] # Simple list for testing
        self.word = ""
        self.lives = 0
        self.masked_word = ""
        self.guessed = []
        self.repeated_guesses_in_episode = 0
        self.wrong_guesses_in_episode = 0

    def reset(self, word_to_play=None):
        """Starts a new game."""
        if word_to_play:
            self.word = word_to_play.upper()
        else:
            # Pick a random word from our small list for training
            self.word = random.choice(self.word_list).upper()
            
        self.lives = 6
        self.masked_word = "_" * len(self.word)
        self.guessed = []
        self.repeated_guesses_in_episode = 0
        self.wrong_guesses_in_episode = 0
        return self.get_state()

    def get_state(self):
        """Returns the current state for the Q-table."""
        wrong_guesses = sorted([g for g in self.guessed if g not in self.word])
        return (self.masked_word, tuple(wrong_guesses))

    def step(self, action):
        """
        Takes an action (letter) and returns the new state,
        reward, and done status.
        """
        action = action.upper()
        
        # Check for invalid or repeated guess
        if action not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or len(action) != 1 or action in self.guessed:
            self.repeated_guesses_in_episode += 1
            # Return (state, reward, done, status)
            return self.get_state(), -2, False, 'repeat' # Repeated guess penalty

        self.guessed.append(action)

        # Check if guess is correct
        if action in self.word:
            new_masked = ""
            for i, char in enumerate(self.word):
                if self.word[i] == action or self.masked_word[i] != '_':
                    new_masked += self.word[i]
                else:
                    new_masked += "_"
            self.masked_word = new_masked

            # Check for WIN
            if self.masked_word == self.word:
                return self.get_state(), 2000, True, 'win'
            else:
                return self.get_state(), 1, False, 'correct'
        else:
            # Guess is wrong
            self.lives -= 1
            self.wrong_guesses_in_episode += 1
            
            # Check for LOSS
            if self.lives == 0:
                return self.get_state(), -2000, True, 'loss'
            else:
                return self.get_state(), -5, False, 'wrong'