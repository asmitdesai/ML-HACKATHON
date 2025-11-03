import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import config
from utils import word_to_one_hot, guessed_to_binary_vec

# --- 1. DQN Network Definition  ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Input state: HMM Probs (26) + Guessed Vec (26) + Lives (1)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size) # Output Q-values for each letter

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. Replay Buffer ---
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# --- 3. The Agent Itself ---
class HangmanAgent:
    def __init__(self, state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-Network 
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LR)
        
        # Replay memory
        self.memory = ReplayBuffer(config.BUFFER_SIZE, config.BATCH_SIZE)
        self.epsilon = config.EPSILON_START

    def state_to_vector(self, state, hmm_probs):
        """
        Converts the state (env dict + HMM dict) into a single
        vector for the neural network[cite: 30, 52].
        """
        # 1. HMM Probabilities (vector)
        probs_vec = np.array([hmm_probs[l] for l in config.ALPHABET])
        
        # 2. Guessed Letters (binary vector)
        guessed_vec = np.array(guessed_to_binary_vec(state['guessed_letters']))
        
        # 3. Lives (scalar)
        lives_vec = np.array([state['lives'] / config.MAX_LIVES])
        
        # Concatenate all parts
        # Note: We are using HMM probs + guessed letters + lives.
        # We've omitted the masked_word one-hot for simplicity,
        # as the HMM probs already encode that information.
        full_state_vec = np.concatenate((probs_vec, guessed_vec, lives_vec))
        return full_state_vec

    def select_action(self, state_vector, guessed_letters):
        """
        Selects an action using Epsilon-Greedy strategy[cite: 55].
        """
        if random.random() < self.epsilon:
            # Exploration: Pick a random *unguessed* letter
            available_actions = [i for i, l in enumerate(config.ALPHABET) 
                                 if l not in guessed_letters]
            if not available_actions:
                return random.randrange(self.action_size) # Failsafe
            return random.choice(available_actions)
        else:
            # Exploitation: Pick the best action from the Q-network
            state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            # Mask out already guessed letters by setting their Q-value to -inf
            for i, l in enumerate(config.ALPHABET):
                if l in guessed_letters:
                    q_values[0][i] = -float('inf')
                    
            return np.argmax(q_values.cpu().data.numpy())

    def optimize_model(self):
        """
        Trains the Q-Network using a batch from the replay buffer.
        """
        if len(self.memory) < config.BATCH_SIZE:
            return # Not enough samples to train yet

        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # Get Q-values for next states from target network
        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (config.GAMMA * Q_targets_next * (1 - dones))
        
        # Get expected Q-values from main network
        Q_expected = self.q_network(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Syncs the target network with the main Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def decay_epsilon(self):
        """Decays epsilon for the Epsilon-Greedy strategy[cite: 55]."""
        self.epsilon = max(config.EPSILON_END, config.EPSILON_DECAY * self.epsilon)
