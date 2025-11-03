# --- 1. MOCK FUNCTIONS (Delete these when you get the real ones) ---
# Mock Asmit's HMM
import random

from q_agent import QLearningAgent 
def get_letter_probabilities(masked_word, guessed_letters):
    return {letter: random.random() for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}

# Mock Aryan's Environment
class HangmanEnv:
    def __init__(self):
        self.lives = 6
        self.word = "PYTHON"
        self.masked_word = "______"
        self.guessed = []

    def reset(self):
        # In the real version, this will pick a word from corpus.txt
        self.lives = 6
        self.masked_word = "______"
        self.guessed = []
        return self.get_state()

    def get_state(self):
        # This is the state definition we discussed
        wrong_guesses = sorted([g for g in self.guessed if g not in self.word])
        return (self.masked_word, tuple(wrong_guesses))

    def step(self, action):
        # --- This is the complex logic Aryan must build ---
        # It must return (new_state, reward, done)
        
        if action in self.guessed:
            # [cite_start]You must design this reward! [cite: 32]
            return self.get_state(), -2, False # Repeated guess penalty
        
        self.guessed.append(action)

        if action in self.word:
            # Update masked_word
            new_masked = ""
            for i, char in enumerate(self.word):
                if char == action:
                    new_masked += action
                else:
                    new_masked += self.masked_word[i]
            self.masked_word = new_masked

            # Check for win
            if self.masked_word == self.word:
                return self.get_state(), 2000, True # Win reward
            else:
                return self.get_state(), 1, False # Correct guess reward
        else:
            self.lives -= 1
            # Check for loss
            if self.lives == 0:
                return self.get_state(), -2000, True # Lose reward
            else:
                return self.get_state(), -5, False # Wrong guess penalty

# --- 2. YOUR MAIN SCRIPT ---

 
# from hmm import get_letter_probabilities  <-- (Asmit's file)
# from environment import HangmanEnv      <-- (Aryan's file)

agent = QLearningAgent()
env = HangmanEnv()
NUM_EPISODES = 10000 # Tune this based on time

print("--- Starting Training ---")

for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    
    while not done:
        # 1. Get HMM's advice
        masked_word, wrong_guesses = state
        guessed_letters = list(wrong_guesses) + [c.lower() for c in masked_word if c != '_']
        hmm_probs = get_letter_probabilities(masked_word, guessed_letters)
        
        # 2. Agent chooses action
        action = agent.choose_action(state, hmm_probs)
        
        # 3. Environment gives result
        new_state, reward, done = env.step(action)
        
        # 4. Agent learns
        agent.learn(state, action, reward, new_state, done)
        
        # 5. Update state for next loop
        state = new_state
        
    # After the game is over, decay epsilon
    agent.decay_epsilon()

    if episode % 100 == 0:
        print(f"Episode: {episode}, Epsilon: {agent.epsilon:.3f}")

print("--- Training Finished ---")

# (Your evaluation code using test.txt will go here)