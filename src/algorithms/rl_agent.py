import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from numba import njit


class CirclePlacementNet(nn.Module):
    """
    Neural network for the DQN agent.
    Takes as input:
    - Current map state (flattened)
    - Current radius to place
    - Number of circles already placed
    
    Outputs Q-values for each possible position on the map.
    """
    
    def __init__(self, map_size, hidden_size=512):
        super(CirclePlacementNet, self).__init__()
        self.map_size = map_size
        self.input_size = map_size * map_size + 2  # flattened map + radius + num_placed
        
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, map_size * map_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x.view(-1, self.map_size, self.map_size)


class CirclePlacementEnv:
    """
    Environment for the circle placement task.
    """
    
    def __init__(self, map_size=64, radii=None):
        self.map_size = map_size
        self.radii = radii if radii is not None else [12, 8, 7, 6, 5, 4, 3, 2, 1]
        self.reset()
        
    def reset(self, weighted_matrix=None):
        """Reset the environment with a new map."""
        if weighted_matrix is None:
            self.original_map = self._generate_random_map()
        else:
            self.original_map = weighted_matrix.copy()
        
        self.current_map = self.original_map.copy()
        self.current_radius_idx = 0
        self.placed_circles = []
        self.total_weight_collected = 0
        return self._get_state()
    
    def _generate_random_map(self, time_steps=10000):
        """Generate a random weighted map using the same method as in main.py"""
        return random_seeder(self.map_size, time_steps)
    
    def _get_state(self):
        """Get the current state representation."""
        # Create state map with clear indication of placed circles
        # Areas with value 0 in current_map but non-zero in original_map were covered by circles
        
        if self.original_map.max() > 0:
            # Normalize current values by original max
            normalized_map = self.current_map / self.original_map.max()
            
            # Mark already placed areas as -1
            # These are areas that had value in original but are now 0
            already_placed = (self.original_map > 0) & (self.current_map == 0)
            
            # Also mark areas that are currently zero but were originally zero as 0
            # to distinguish from high-value areas
            originally_zero = (self.original_map == 0)
            
            state_map = normalized_map.copy()
            state_map[already_placed] = -1.0  # Already covered by circles
            state_map[originally_zero] = 0.0  # Originally empty areas
        else:
            state_map = self.current_map
        
        # Current radius (normalized)
        current_radius = self.radii[self.current_radius_idx] / max(self.radii)
        
        # Number of circles placed (normalized)
        num_placed = self.current_radius_idx / len(self.radii)
        
        # Flatten and concatenate
        state = np.concatenate([
            state_map.flatten(),
            [current_radius, num_placed]
        ])
        
        return state
    
    def step(self, action):
        """
        Take a step in the environment.
        Action is a tuple (x, y) representing where to place the circle.
        """
        x, y = action
        radius = self.radii[self.current_radius_idx]
        
        # Calculate reward as the weighted area covered
        included_weight = compute_included(self.current_map, x, y, radius)
        
        # Normalize reward
        # If we collected nothing (complete overlap), give large negative reward
        if included_weight <= 0:
            reward = -1000  # Large negative for baseline agent
        else:
            # Reward is the actual weight collected
            reward = included_weight
        
        # Store the placement
        self.placed_circles.append((x, y, radius))
        self.total_weight_collected += included_weight
        
        # Move to next radius
        self.current_radius_idx += 1
        
        # Check if done
        done = self.current_radius_idx >= len(self.radii)
        
        # Get next state
        next_state = self._get_state() if not done else None
        
        # Additional info
        info = {
            'total_weight': np.sum(self.original_map),
            'weight_collected': self.total_weight_collected,
            'coverage_ratio': self.total_weight_collected / np.sum(self.original_map) if np.sum(self.original_map) > 0 else 0
        }
        
        return next_state, reward, done, info
    
    def get_valid_actions_mask(self):
        """
        Get a mask of valid actions (positions where we can place the current circle).
        This helps avoid placing circles outside the map boundaries AND on existing circles.
        """
        radius = self.radii[self.current_radius_idx]
        mask = np.ones((self.map_size, self.map_size), dtype=bool)
        
        # Mark positions too close to boundaries as invalid
        mask[:int(radius), :] = False
        mask[-int(radius):, :] = False
        mask[:, :int(radius)] = False
        mask[:, -int(radius):] = False
        
        # CRITICAL: Mark positions that would overlap with existing circles as invalid
        for px, py, pr in self.placed_circles:
            # Calculate minimum distance needed to avoid overlap
            min_distance = radius + pr
            
            # Mark all positions within overlap distance as invalid
            for i in range(max(0, int(px - min_distance)), min(self.map_size, int(px + min_distance + 1))):
                for j in range(max(0, int(py - min_distance)), min(self.map_size, int(py + min_distance + 1))):
                    # Check if this position would cause overlap
                    dist = np.sqrt((i - px)**2 + (j - py)**2)
                    if dist < min_distance:
                        mask[i, j] = False
        
        # Also mark positions where there's no value to collect as less desirable
        # (but still valid in case everything else is blocked)
        no_value_mask = (self.current_map == 0)
        
        # If we have any positions with value, prefer those
        if np.any(mask & (self.current_map > 0)):
            # Invalidate zero-value positions if we have better options
            mask = mask & (self.current_map > 0)
        
        return mask


class DQNAgent:
    """
    Deep Q-Learning agent for circle placement.
    """
    
    def __init__(self, map_size=64, radii=None, learning_rate=1e-4, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay=0.995, buffer_size=10000, batch_size=32):
        
        self.map_size = map_size
        self.radii = radii if radii is not None else [12, 8, 7, 6, 5, 4, 3, 2, 1]
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Neural networks
        self.q_network = CirclePlacementNet(map_size)
        self.target_network = CirclePlacementNet(map_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Training metrics
        self.losses = []
        self.rewards = []
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_mask=None):
        """
        Choose an action using epsilon-greedy policy.
        """
        if random.random() < self.epsilon:
            # Random valid action
            if valid_mask is not None:
                valid_positions = np.argwhere(valid_mask)
                if len(valid_positions) > 0:
                    idx = random.randint(0, len(valid_positions) - 1)
                    return tuple(valid_positions[idx])
            # Fallback to completely random
            x = random.randint(0, self.map_size - 1)
            y = random.randint(0, self.map_size - 1)
            return (x, y)
        else:
            # Greedy action from Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).squeeze(0)
            
            # Apply valid mask if provided
            if valid_mask is not None:
                q_values_np = q_values.numpy()
                q_values_np[~valid_mask] = -np.inf
                best_idx = np.unravel_index(np.argmax(q_values_np), q_values_np.shape)
            else:
                best_idx = np.unravel_index(torch.argmax(q_values).item(), q_values.shape)
            
            return best_idx
    
    def replay(self):
        """Train the Q-network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = [e[3] for e in batch]
        dones = torch.FloatTensor([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.view(self.batch_size, -1)
        action_indices = actions[:, 0] * self.map_size + actions[:, 1]
        current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        next_q_values = torch.zeros(self.batch_size)
        non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)
        
        if non_final_mask.any():
            non_final_next_states = torch.FloatTensor([s for s in next_states if s is not None])
            with torch.no_grad():
                next_q_values[non_final_mask] = self.target_network(non_final_next_states).view(
                    non_final_mask.sum(), -1).max(1)[0]
        
        # Compute targets
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# Copy the njit function from main.py
@njit()
def random_seeder(dim, time_steps=10000):
    x = np.random.uniform(0, 1, (dim, dim))
    seed_pos_x = int(np.random.uniform(0, dim))
    seed_pos_y = int(np.random.uniform(0, dim))
    tele_prob = 0.001
    for i in range(time_steps):
        x[seed_pos_x, seed_pos_y] += np.random.uniform(0, 1)
        if np.random.uniform() < tele_prob:
            seed_pos_x = int(np.random.uniform(0, dim))
            seed_pos_y = int(np.random.uniform(0, dim))
        else:
            if np.random.uniform() < 0.5:
                seed_pos_x += 1
            if np.random.uniform() < 0.5:
                seed_pos_x += -1
            if np.random.uniform() < 0.5:
                seed_pos_y += 1
            if np.random.uniform() < 0.5:
                seed_pos_y += -1
            seed_pos_x = int(max(min(seed_pos_x, dim - 1), 0))
            seed_pos_y = int(max(min(seed_pos_y, dim - 1), 0))
    return x


# Copy the compute_included function from loss_functions.py
@njit()
def compute_included(weighted_matrix, center_x, center_y, radius):
    """
    Compute the weighted sum of values in a matrix covered by a circle.
    Each matrix cell is treated as a continuous area with a uniformly distributed weight.
    The function updates the matrix in-place (sets weights to zero after including them).
    """
    H, W = weighted_matrix.shape
    start_x = max(int(np.floor(center_x - radius)), 0)
    end_x = min(int(np.ceil(center_x + radius)) + 1, H)
    start_y = max(int(np.floor(center_y - radius)), 0)
    end_y = min(int(np.ceil(center_y + radius)) + 1, W)

    included_weight = 0.0
    r2 = radius * radius

    for i in range(start_x, end_x):
        for j in range(start_y, end_y):
            dx = i + 0.5 - center_x
            dy = j + 0.5 - center_y
            if dx * dx + dy * dy <= r2:
                included_weight += weighted_matrix[i, j]
                weighted_matrix[i, j] = 0  # mark as consumed
    return included_weight