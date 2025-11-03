import random
from collections import defaultdict
from main import guessed_letters

class QLearningAgent:
    """
    This is the "brain" of your hackathon project.
    It uses a Q-table to learn strategy and combines it
    with the HMM's probabilities to pick the best letter.
    """
    def __init__(self, 
                 learning_rate=0.1, 
                 discount_factor=0.9,
                 epsilon=1.0, 
                 epsilon_decay=0.999, 
                 min_epsilon=0.01,
                 hmm_weight=50.0):
        
        # --- Hyperparameters ---
        self.lr = learning_rate         # Alpha: How fast the agent learns
        self.gamma = discount_factor    # Gamma: How much it values future rewards
        self.epsilon = epsilon          # Exploration rate: 1.0 = 100% random
        self.epsilon_decay = epsilon_decay  # How fast epsilon shrinks
        self.min_epsilon = min_epsilon    # The lowest epsilon will go
        
        # [cite_start]This is the "hybrid" part[cite: 14]. 
        # How much should we value the HMM's advice?
        self.HMM_WEIGHT = hmm_weight    
        
        # --- Q-Table ---
        # A dictionary where: Q[state][action] -> learned_value
        # e.g., Q["('P____N', ('A', 'E'))"]["L"] -> 4.5
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # [cite_start]The 26 possible actions (letters to guess) [cite: 31]
        self.actions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def choose_action(self, state, hmm_probs):
        """
        Chooses an action using an epsilon-greedy policy,
        hybridized with HMM probabilities.
        """
        # --- 2. Create a list of valid, remaining actions ---
        remaining_actions = [l for l in self.actions if l.lower() not in guessed_letters]
    
    # Failsafe: If no actions are left (shouldn't happen), guess 'A'
        if not remaining_actions:
            return "A"
        # --- 1. Exploration ---
        # If a random number is less than epsilon, pick a random letter.
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # --- 2. Exploitation ---
        # Otherwise, pick the best-known action.
        else:
            best_action = None
            max_score = -float('inf')
            
            # Iterate through every possible letter
            for action in self.actions:
                # Get the agent's learned strategic value for this action
                q_value = self.q_table[state].get(action, 0.0)
                
                # Get the HMM's "oracle" advice for this action
                hmm_value = hmm_probs.get(action, 0.0)
                
                # --- The Hybrid Score ---
                # Combine learned strategy (Q) with HMM's advice
                hybrid_score = q_value + (hmm_value * self.HMM_WEIGHT)
                
                if hybrid_score > max_score:
                    max_score = hybrid_score
                    best_action = action
                    
            return best_action if best_action else random.choice(remaining_actions)

    def learn(self, state, action, reward, next_state, done):
        """
        Updates the Q-table using the Bellman equation after each step.
        """
        
        # Find the max Q-value for the *next* state
        max_q_next = 0.0
        if not done:
            # This is the fix:
            # Check if the dictionary for next_state is NOT empty
            if self.q_table[next_state]: 
                max_q_next = max(self.q_table[next_state].values())
        
        # The Q-learning update rule
        current_q = self.q_table[state].get(action, 0.0)
        td_target = reward + self.gamma * max_q_next
        td_error = td_target - current_q
        
        # Update the Q-value for the state-action pair
        self.q_table[state][action] = current_q + self.lr * td_error

    def decay_epsilon(self):
        """Call this at the end of each episode (game) to reduce randomness."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)