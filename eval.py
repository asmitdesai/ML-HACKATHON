import sys
from collections import defaultdict
import random

# --- Import Your Team's Files ---
# Make sure these files are in the same directory

# from q_agent import QLearningAgent   Your QLearningAgent class
# from environment import HangmanEnv      # Aryan's Environment class
# from hmm import get_letter_probabilities  # Asmit's HMM function

# --- MOCK FUNCTIONS (Delete these when you have the real ones) ---
# Mock HMM
def get_letter_probabilities(masked_word, guessed_letters):
    return {letter: random.random() for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}

# Mock Agent (This MUST be replaced with your real, trained agent)
class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.actions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.HMM_WEIGHT = 50.0 # Make sure this matches your trained agent's weight
        self.epsilon = 0.0 # Set epsilon to 0 for evaluation!

    def choose_action(self, state, hmm_probs):
        # Evaluation mode: NO random actions (epsilon = 0)
        best_action = None
        max_score = -float('inf')
        for action in self.actions:
            q_value = self.q_table[state].get(action, 0.0)
            hmm_value = hmm_probs.get(action, 0.0)
            hybrid_score = q_value + (hmm_value * self.HMM_WEIGHT)
            if hybrid_score > max_score:
                max_score = hybrid_score
                best_action = action
        return best_action

# Mock Environment (This MUST be replaced with Aryan's real class)
class HangmanEnv:
    def __init__(self, word_list):
        self.word_list = word_list
        self.word = ""
        self.lives = 0
        self.masked_word = ""
        self.guessed = []
        self.repeated_guesses_in_episode = 0
        self.wrong_guesses_in_episode = 0

    def reset(self, word_to_play):
        self.word = word_to_play.upper()
        self.lives = 6
        self.masked_word = "_" * len(self.word)
        self.guessed = []
        self.repeated_guesses_in_episode = 0
        self.wrong_guesses_in_episode = 0
        return self.get_state()

    def get_state(self):
        wrong_guesses = sorted([g for g in self.guessed if g not in self.word])
        return (self.masked_word, tuple(wrong_guesses))

    def step(self, action):
        action = action.upper()
        
        if action not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or len(action) != 1:
            # Invalid action, treat as a repeat
            self.repeated_guesses_in_episode += 1
            return self.get_state(), -2, False, 'repeat' # Repeated guess penalty

        if action in self.guessed:
            self.repeated_guesses_in_episode += 1
            return self.get_state(), -2, False, 'repeat' # Repeated guess penalty
        
        self.guessed.append(action)

        if action in self.word:
            new_masked = ""
            for i, char in enumerate(self.word):
                if char == action or self.masked_word[i] != '_':
                    new_masked += self.word[i]
                else:
                    new_masked += "_"
            self.masked_word = new_masked

            if self.masked_word == self.word:
                return self.get_state(), 2000, True, 'win'
            else:
                return self.get_state(), 1, False, 'correct'
        else:
            self.lives -= 1
            self.wrong_guesses_in_episode += 1
            if self.lives == 0:
                return self.get_state(), -2000, True, 'loss'
            else:
                return self.get_state(), -5, False, 'wrong'

# --- END MOCK FUNCTIONS ---


def load_test_words(filename="test.txt"):
    """Loads the test.txt file."""
    try:
        with open(filename, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        if len(words) < 2000:
            print(f"Warning: {filename} contains {len(words)} words, not 2000.")
            # Use all available words if less than 2000
            return words
        # [cite_start]As per the PDF, it will be evaluated on 2000 games [cite: 41]
        # We assume the file *is* the test set.
        return words[:2000] 
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        print("Please make sure 'test.txt' is in the same folder.")
        sys.exit(1)


# --- 1. SETUP ---

# !!! IMPORTANT !!!
# Load your TRAINED agent here.
# If you saved it as a file (e.g., pickle), you must load it.
# For this example, we just create a new agent.
#
# agent = QLearningAgent() 
# agent.load_q_table('my_trained_q_table.pkl') # <-- Example
#
# For an 8-hour hackathon, you might just have the agent
# in memory from your training script.
#
# agent = my_trained_agent_from_previous_script 
# agent.epsilon = 0.0 # CRITICAL: Set exploration to ZERO

agent = QLearningAgent() # Mock: Replace with your trained agent
agent.epsilon = 0.0 # Set to 0 for pure exploitation

print("Loading test words from test.txt...")

test_words = load_test_words("test.txt") # [cite: 93]
num_games = len(test_words)

env = HangmanEnv(test_words) # Aryan's class, initialized with test words

# --- 2. EVALUATION LOOP ---

total_wins = 0
total_wrong_guesses = 0
total_repeated_guesses = 0

print(f"--- Running Evaluation on {num_games} games ---")

for i in range(num_games):
    word_to_play = test_words[i]
    state = env.reset(word_to_play)
    done = False
    
    while not done:
        # 1. Get HMM's advice
        masked_word, wrong_guesses = state
        guessed_letters = list(wrong_guesses) + [c for c in masked_word if c != '_']
        hmm_probs = get_letter_probabilities(masked_word, guessed_letters) # Asmit's function
        
        # 2. Agent chooses action (NO EXPLORATION)
        action = agent.choose_action(state, hmm_probs)
        
        # 3. Environment gives result
        new_state, reward, done, game_status = env.step(action)
        
        # 4. Update state
        state = new_state
        
        # 5. Check for win
        if done and game_status == 'win':
            total_wins += 1

    # After the game, add this episode's stats to the total
    total_wrong_guesses += env.wrong_guesses_in_episode
    total_repeated_guesses += env.repeated_guesses_in_episode
    
    if (i + 1) % 200 == 0:
        print(f"Played {i+1} / {num_games} games...")

print("--- Evaluation Finished ---")

# --- 3. FINAL SCORE CALCULATION ---

# [cite_start]Note: The problem statement [cite: 44] says "1,000 games" in the 
# [cite_start]Success Rate definition, but "2000 games" in the main paragraph[cite: 41].
# We will use the 2000 number from the main paragraph.
if num_games == 0:
    print("No games played. Exiting.")
    sys.exit(0)

success_rate = total_wins / num_games

# [cite_start]The formula from the PDF [cite: 43]
# Final Score = (Success Rate * 2000) - (Total Wrong Guesses * 5) - (Total Repeated Guesses * 2)

score_from_wins = success_rate * 2000
penalty_from_wrong = total_wrong_guesses * 5
penalty_from_repeats = total_repeated_guesses * 2

final_score = score_from_wins - penalty_from_wrong - penalty_from_repeats

# --- 4. DISPLAY RESULTS ---

print("\n--- HACKMAN FINAL RESULTS ---")
print(f"Games Played:           {num_games}")
print("---------------------------------")
print(f"Total Wins:             {total_wins} ({success_rate * 100:.2f}%)")
print(f"Total Wrong Guesses:    {total_wrong_guesses}")
print(f"Total Repeated Guesses: {total_repeated_guesses}")
print("---------------------------------")
print(f"Score from Wins:        + {score_from_wins:.2f}")
print(f"Penalty (Wrong):        - {penalty_from_wrong}")
print(f"Penalty (Repeats):      - {penalty_from_repeats}")
print("---------------------------------")
print(f"FINAL SCORE:            {final_score:.2f}")