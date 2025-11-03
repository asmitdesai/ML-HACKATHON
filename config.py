# --- Constants ---
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
MAX_LIVES = 6  # As specified in the evaluation

# --- File Paths ---
CORPUS_PATH = 'corpus.txt'
HMM_MODEL_PATH = 'hmm_model.pkl'  # To save your trained HMM
AGENT_MODEL_PATH = 'agent_model.pth' # To save your trained RL Agent

# --- HMM Parameters ---
HMM_N_STATES = 10  # Example: Number of hidden states (you must tune this)

# --- RL (DQN) Hyperparameters ---
STATE_SIZE = 26 + 26 + 1  # (Masked word one-hot + guessed letters binary vec + lives)
ACTION_SIZE = 26          # 26 letters in the alphabet

LR = 1e-4                 # Learning Rate
BATCH_SIZE = 64           # How many experiences to sample from memory
GAMMA = 0.99              # Discount factor for future rewards

# --- Replay Buffer ---
BUFFER_SIZE = 100000      # Max size of the replay memory

# --- Exploration (Epsilon-Greedy) ---
EPSILON_START = 1.0       # Start with 100% exploration
EPSILON_END = 0.01        # End with 1% exploration
EPSILON_DECAY = 0.995     # Multiplicative decay rate

# --- Training & Evaluation ---
NUM_EPISODES_TRAIN = 10000   # Number of games to train the agent on
NUM_EPISODES_EVAL = 2000     # Number of games for final evaluation
