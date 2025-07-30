import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from numba import njit
import multiprocessing as mp
from torch.nn.utils import clip_grad_norm_
import math


class ImprovedCirclePlacementNet(nn.Module):
    """
    Enhanced neural network with attention mechanisms and better architecture.
    """
    
    def __init__(self, map_size, hidden_size=512, num_heads=4):
        super(ImprovedCirclePlacementNet, self).__init__()
        self.map_size = map_size
        self.input_size = map_size * map_size + 2
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size after convolutions
        conv_out_size = (map_size // 4) * (map_size // 4) * 128
        
        # Global context vector (radius and num_placed)
        self.context_fc = nn.Linear(2, 128)
        
        # Simplified architecture without attention for memory efficiency
        self.fc1 = nn.Linear(conv_out_size + 128, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, map_size * map_size)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Split input into map and context
        map_input = x[:, :-2].view(batch_size, 1, self.map_size, self.map_size)
        context_input = x[:, -2:]
        
        # Convolutional feature extraction with pooling
        conv_out = F.relu(self.bn1(self.conv1(map_input)))
        conv_out = F.relu(self.bn2(self.conv2(conv_out)))
        conv_out = F.relu(self.bn3(self.conv3(conv_out)))
        
        # Flatten convolutional output
        conv_flat = conv_out.view(batch_size, -1)
        
        # Process context
        context_features = F.relu(self.context_fc(context_input))
        
        # Combine features
        combined = torch.cat([conv_flat, context_features], dim=1)
        
        # Final layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x.view(batch_size, self.map_size, self.map_size)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer for more efficient learning.
    """
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def __len__(self):
        return len(self.buffer)


class ParallelEnvWorker(mp.Process):
    """
    Worker process for parallel environment simulation.
    """
    
    def __init__(self, worker_id, map_size, radii, input_queue, output_queue):
        super().__init__()
        self.worker_id = worker_id
        self.map_size = map_size
        self.radii = radii
        self.input_queue = input_queue
        self.output_queue = output_queue
        
    def run(self):
        # Create environment for this worker
        env = CirclePlacementEnv(self.map_size, self.radii)
        
        while True:
            cmd, data = self.input_queue.get()
            
            if cmd == 'reset':
                state = env.reset()
                self.output_queue.put((self.worker_id, state))
                
            elif cmd == 'step':
                action = data
                next_state, reward, done, info = env.step(action)
                self.output_queue.put((self.worker_id, (next_state, reward, done, info)))
                
            elif cmd == 'get_valid_mask':
                mask = env.get_valid_actions_mask()
                self.output_queue.put((self.worker_id, mask))
                
            elif cmd == 'close':
                break


class ImprovedDQNAgent:
    """
    Enhanced DQN agent with advanced features.
    """
    
    def __init__(self, map_size=64, radii=None, learning_rate=3e-4, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay_steps=50000, buffer_size=100000, 
                 batch_size=32, tau=0.005, n_step=3, 
                 use_double_dqn=True, use_dueling=True, device=None):
        
        self.map_size = map_size
        self.radii = radii if radii is not None else [12, 8, 7, 6, 5, 4, 3, 2, 1]
        self.gamma = gamma
        self.n_step = n_step
        self.tau = tau  # Soft update parameter
        self.use_double_dqn = use_double_dqn
        self.batch_size = batch_size
        
        # Epsilon scheduling
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.steps_done = 0
        
        # Device configuration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Neural networks
        self.q_network = ImprovedCirclePlacementNet(map_size).to(self.device)
        self.target_network = ImprovedCirclePlacementNet(map_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer with learning rate scheduling
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100000, eta_min=learning_rate/10
        )
        
        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(buffer_size)
        
        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Training metrics
        self.losses = []
        self.rewards = []
        self.q_values = []
        self.gradient_norms = []
        
    def update_epsilon(self):
        """Update epsilon using exponential decay."""
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      math.exp(-1. * self.steps_done / self.epsilon_decay_steps)
        self.steps_done += 1
        
    def act(self, state, valid_mask=None):
        """Choose action using epsilon-greedy policy."""
        self.update_epsilon()
        
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
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).squeeze(0).cpu()
                self.q_values.append(q_values.max().item())
            
            # Apply valid mask if provided
            if valid_mask is not None:
                q_values_np = q_values.numpy()
                q_values_np[~valid_mask] = -np.inf
                best_idx = np.unravel_index(np.argmax(q_values_np), q_values_np.shape)
            else:
                best_idx = np.unravel_index(torch.argmax(q_values).item(), q_values.shape)
            
            return best_idx
    
    def remember_n_step(self, state, action, reward, next_state, done):
        """Store experience in n-step buffer."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step or done:
            # Calculate n-step return
            n_step_return = 0
            for i in range(len(self.n_step_buffer)):
                n_step_return += (self.gamma ** i) * self.n_step_buffer[i][2]
            
            # Get the first and last states
            first_state = self.n_step_buffer[0][0]
            first_action = self.n_step_buffer[0][1]
            last_state = self.n_step_buffer[-1][3]
            last_done = self.n_step_buffer[-1][4]
            
            # Store in replay buffer
            self.memory.push(first_state, first_action, n_step_return, last_state, last_done)
            
            # Clear buffer if episode is done
            if done:
                self.n_step_buffer.clear()
    
    def replay(self):
        """Train the Q-network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        batch, indices, weights = self.memory.sample(self.batch_size)
        
        # Convert batch to numpy arrays first for efficiency
        states_np = np.array([e[0] for e in batch])
        actions_np = np.array([e[1] for e in batch])
        rewards_np = np.array([e[2] for e in batch])
        next_states = [e[3] for e in batch]
        dones_np = np.array([e[4] for e in batch])
        
        # Convert to tensors
        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor(actions_np).to(self.device)
        rewards = torch.FloatTensor(rewards_np).to(self.device)
        dones = torch.FloatTensor(dones_np).to(self.device)
        weights = weights.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.view(self.batch_size, -1)
        action_indices = actions[:, 0] * self.map_size + actions[:, 1]
        current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        next_q_values = torch.zeros(self.batch_size).to(self.device)
        non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool).to(self.device)
        
        if non_final_mask.any():
            non_final_next_states_np = np.array([s for s in next_states if s is not None])
            non_final_next_states = torch.FloatTensor(non_final_next_states_np).to(self.device)
            
            if self.use_double_dqn:
                # Double DQN: use online network to select action, target network to evaluate
                with torch.no_grad():
                    next_q_online = self.q_network(non_final_next_states).view(
                        non_final_mask.sum(), -1)
                    next_actions = next_q_online.max(1)[1]
                    next_q_target = self.target_network(non_final_next_states).view(
                        non_final_mask.sum(), -1)
                    next_q_values[non_final_mask] = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                with torch.no_grad():
                    next_q_values[non_final_mask] = self.target_network(non_final_next_states).view(
                        non_final_mask.sum(), -1).max(1)[0]
        
        # Compute targets
        targets = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values
        
        # Compute TD errors for prioritized replay
        td_errors = torch.abs(current_q_values - targets).detach()
        
        # Weighted loss
        loss = (weights * F.smooth_l1_loss(current_q_values, targets, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.gradient_norms.append(grad_norm.item())
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors.cpu().numpy() + 1e-6)
        
        self.losses.append(loss.item())
        
        # Soft update target network
        self.soft_update_target_network()
    
    def soft_update_target_network(self):
        """Soft update of target network parameters."""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def get_statistics(self):
        """Get current training statistics."""
        stats = {
            'epsilon': self.epsilon,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_q_value': np.mean(self.q_values[-100:]) if self.q_values else 0,
            'avg_gradient_norm': np.mean(self.gradient_norms[-100:]) if self.gradient_norms else 0,
            'buffer_size': len(self.memory)
        }
        return stats


# Keep the original environment and njit functions
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
        # Normalize the map to [0, 1]
        if self.current_map.max() > 0:
            normalized_map = self.current_map / self.current_map.max()
        else:
            normalized_map = self.current_map
        
        # Current radius (normalized)
        current_radius = self.radii[self.current_radius_idx] / max(self.radii)
        
        # Number of circles placed (normalized)
        num_placed = self.current_radius_idx / len(self.radii)
        
        # Flatten and concatenate
        state = np.concatenate([
            normalized_map.flatten(),
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
        weight_before = np.sum(self.current_map)
        included_weight = compute_included(self.current_map, x, y, radius)
        weight_after = np.sum(self.current_map)
        
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
        This helps avoid placing circles outside the map boundaries.
        """
        radius = self.radii[self.current_radius_idx]
        mask = np.ones((self.map_size, self.map_size), dtype=bool)
        
        # Mark positions too close to boundaries as invalid
        mask[:int(radius), :] = False
        mask[-int(radius):, :] = False
        mask[:, :int(radius)] = False
        mask[:, -int(radius):] = False
        
        return mask


# Copy the njit functions
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


@njit()
def compute_included(weighted_matrix, center_x, center_y, radius):
    """
    Compute the weighted sum of values in a matrix covered by a circle.
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
                if weighted_matrix[i, j] == 0:
                    included_weight += -1
                weighted_matrix[i, j] = 0  # mark as consumed
    return included_weight