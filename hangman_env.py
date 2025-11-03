import random
import config
from utils import mask_word

class HangmanEnv:
    """
    Implements the Hangman Game Environment for the RL agent.
    """
    def __init__(self, word_list):
        self.word_list = word_list
        self.max_lives = config.MAX_LIVES
        self.word = ""
        self.guessed_letters = set()
        self.lives = 0
        
    def _get_state(self):
        """
        Returns the current state of the environment[cite: 29, 30].
        """
        return {
            'masked_word': mask_word(self.word, self.guessed_letters),
            'guessed_letters': self.guessed_letters,
            'lives': self.lives
        }
        
    def reset(self):
        """
        Starts a new game.
        Returns: The initial state dictionary.
        """
        self.word = random.choice(self.word_list)
        self.guessed_letters = set()
        self.lives = self.max_lives
        return self._get_state()

    def step(self, action):
        """
        Performs one action (guessing a letter) in the environment[cite: 31].
        
        Returns:
            - next_state (dict): The state after the action.
            - reward (float): The reward for the action[cite: 32].
            - done (bool): True if the game is over (win or loss).
            - info (dict): Additional info (win/loss, repeated_guess).
        """
        action = action.lower()
        
        # --- Initialize return values ---
        reward = 0.0
        done = False
        info = {
            'wrong_guess': False,
            'repeated_guess': False,
            'win': False
        }

        # --- Check for repeated guess ---
        if action not in config.ALPHABET:
            # Invalid action (not a letter)
            reward = -2  # Penalize invalid action
            done = True
            info['repeated_guess'] = True
        elif action in self.guessed_letters:
            # Repeated guess [cite: 34]
            reward = -2  # Penalize repeated guess
            done = True # We will end the episode on an inefficient guess
            info['repeated_guess'] = True
        else:
            self.guessed_letters.add(action)
            
            # --- Check if guess is correct ---
            if action in self.word:
                # Correct guess
                reward = 1
            else:
                # Wrong guess [cite: 34]
                self.lives -= 1
                reward = -1 # Penalize wrong guess
                info['wrong_guess'] = True

            # --- Check for game end conditions ---
            current_masked_word = mask_word(self.word, self.guessed_letters)
            
            if '_' not in current_masked_word:
                # Win [cite: 33]
                reward = 10  # High reward for winning
                done = True
                info['win'] = True
            elif self.lives <= 0:
                # Loss
                reward = -5  # Penalty for losing
                done = True

        return self._get_state(), reward, done, info
