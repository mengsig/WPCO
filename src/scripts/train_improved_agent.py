"""
Improved RL agent training with better reward shaping for higher coverage.
Key improvements:
1. Correct random_seeder for good maps
2. Better reward function that incentivizes high coverage
3. Tuned hyperparameters
"""

import sys
sys.path.append('src')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit


# Correct random_seeder
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


class ImprovedCirclePlacementEnv:
    """Environment with better reward shaping."""
    
    def __init__(self, map_size=64, radii=None):
        self.map_size = map_size
        self.radii = radii if radii is not None else [12, 8, 7, 6, 5, 4, 3, 2, 1]
        self.reset()
        
    def reset(self, weighted_matrix=None):
        if weighted_matrix is None:
            self.original_map = random_seeder(self.map_size, time_steps=100000)
        else:
            self.original_map = weighted_matrix.copy()
        
        self.current_map = self.original_map.copy()
        self.current_radius_idx = 0
        self.placed_circles = []
        self.total_weight_collected = 0
        self.step_count = 0
        
        # Track coverage progression for reward shaping
        self.coverage_history = []
        
        return self._get_state()
    
    def _get_state(self):
        """Enhanced state representation."""
        # Normalize current map
        if self.original_map.max() > 0:
            normalized_map = self.current_map / self.original_map.max()
            
            # Mark placed areas clearly
            already_placed = (self.original_map > 0) & (self.current_map == 0)
            normalized_map[already_placed] = -1.0
        else:
            normalized_map = self.current_map
        
        # Current radius info
        current_radius = self.radii[self.current_radius_idx] / max(self.radii)
        progress = self.current_radius_idx / len(self.radii)
        
        # Add coverage so far
        current_coverage = 1 - (self.current_map.sum() / self.original_map.sum())
        
        # Flatten and concatenate
        state = np.concatenate([
            normalized_map.flatten(),
            [current_radius, progress, current_coverage]
        ])
        
        return state
    
    def step(self, action):
        """Take action with improved reward function."""
        x, y = action
        radius = self.radii[self.current_radius_idx]
        self.step_count += 1
        
        # Calculate weight collected
        included_weight = 0.0
        cells_covered = 0
        for i in range(max(0, int(x - radius)), min(self.map_size, int(x + radius + 1))):
            for j in range(max(0, int(y - radius)), min(self.map_size, int(y + radius + 1))):
                if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                    included_weight += self.current_map[i, j]
                    if self.current_map[i, j] > 0:
                        cells_covered += 1
                    self.current_map[i, j] = 0
        
        self.total_weight_collected += included_weight
        self.placed_circles.append((x, y, radius))
        
        # Calculate coverage
        coverage = 1 - (self.current_map.sum() / self.original_map.sum())
        self.coverage_history.append(coverage)
        
        # IMPROVED REWARD FUNCTION
        # 1. Immediate reward based on weight density
        circle_area = np.pi * radius * radius
        density_reward = included_weight / circle_area if circle_area > 0 else 0
        
        # 2. Coverage improvement reward (key for learning)
        if len(self.coverage_history) > 1:
            coverage_improvement = coverage - self.coverage_history[-2]
        else:
            coverage_improvement = coverage
        
        # 3. Efficiency bonus for large circles
        if radius >= 8:  # Large circles
            efficiency_bonus = density_reward * 2.0  # Double reward for efficient large circles
        elif radius >= 5:  # Medium circles
            efficiency_bonus = density_reward * 1.5
        else:  # Small circles
            efficiency_bonus = density_reward
        
        # 4. Progressive reward based on total coverage
        if coverage > 0.7:
            progress_bonus = 2.0
        elif coverage > 0.5:
            progress_bonus = 1.0
        elif coverage > 0.3:
            progress_bonus = 0.5
        else:
            progress_bonus = 0.0
        
        # Combine rewards
        reward = (
            efficiency_bonus * 10 +  # Scaled efficiency
            coverage_improvement * 100 +  # Strong signal for improvement
            progress_bonus  # Bonus for high coverage
        )
        
        # Move to next radius
        self.current_radius_idx += 1
        done = self.current_radius_idx >= len(self.radii)
        
        # Final bonus
        if done:
            if coverage > 0.8:
                reward += 50
            elif coverage > 0.7:
                reward += 30
            elif coverage > 0.6:
                reward += 20
            elif coverage > 0.5:
                reward += 10
        
        info = {
            'coverage': coverage,
            'included_weight': included_weight,
            'density': density_reward
        }
        
        return self._get_state(), reward, done, info
    
    def get_valid_actions_mask(self):
        """Get valid positions (no overlap + prefer high value areas)."""
        radius = self.radii[self.current_radius_idx]
        mask = np.ones((self.map_size, self.map_size), dtype=float)
        
        # Boundaries
        mask[:int(radius), :] = 0
        mask[-int(radius):, :] = 0
        mask[:, :int(radius)] = 0
        mask[:, -int(radius):] = 0
        
        # No overlap
        for px, py, pr in self.placed_circles:
            min_distance = radius + pr
            for i in range(max(0, int(px - min_distance)), min(self.map_size, int(px + min_distance + 1))):
                for j in range(max(0, int(py - min_distance)), min(self.map_size, int(py + min_distance + 1))):
                    dist = np.sqrt((i - px)**2 + (j - py)**2)
                    if dist < min_distance:
                        mask[i, j] = 0
        
        # Prefer high-value areas by weighting the mask
        value_weight = np.zeros_like(mask)
        for x in range(self.map_size):
            for y in range(self.map_size):
                if mask[x, y] > 0:
                    # Calculate potential value at this position
                    potential = 0
                    for i in range(max(0, x-radius), min(self.map_size, x+radius+1)):
                        for j in range(max(0, y-radius), min(self.map_size, y+radius+1)):
                            if (i-x)**2 + (j-y)**2 <= radius**2:
                                potential += self.current_map[i, j]
                    value_weight[x, y] = potential
        
        # Normalize and combine
        if value_weight.max() > 0:
            value_weight = value_weight / value_weight.max()
            mask = mask * (0.1 + 0.9 * value_weight)  # Minimum 10% probability for exploration
        
        return mask


class DuelingDQN(nn.Module):
    """Dueling DQN architecture for better value estimation."""
    
    def __init__(self, input_size, output_size, hidden_size=512):
        super(DuelingDQN, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Value stream
        self.value_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.value_out = nn.Linear(hidden_size // 2, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.advantage_out = nn.Linear(hidden_size // 2, output_size)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Shared network
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value_out(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage)
        
        # Combine (with advantage centering)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class ImprovedDQNAgent:
    """DQN agent with improvements for better coverage."""
    
    def __init__(self, state_size, action_size, learning_rate=1e-4, gamma=0.99,
                 epsilon_start=0.5, epsilon_end=0.01, epsilon_decay_steps=2000,
                 batch_size=64, buffer_size=100000, update_every=4):
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks
        self.q_network = DuelingDQN(state_size, action_size).to(self.device)
        self.target_network = DuelingDQN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_every = update_every
        self.t_step = 0
        
        # Epsilon
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps_done = 0
        
    def act(self, state, valid_mask=None):
        """Choose action using epsilon-greedy policy."""
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      math.exp(-1. * self.steps_done / self.epsilon_decay_steps)
        
        if random.random() < self.epsilon:
            # Weighted random based on valid mask
            if valid_mask is not None:
                # Sample based on mask weights
                mask_flat = valid_mask.flatten()
                if mask_flat.sum() > 0:
                    probs = mask_flat / mask_flat.sum()
                    action_idx = np.random.choice(len(mask_flat), p=probs)
                    return (action_idx // 64, action_idx % 64)
            return (random.randint(0, 63), random.randint(0, 63))
        
        # Greedy action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
        
        # Apply mask
        if valid_mask is not None:
            q_values = q_values.reshape(64, 64)
            q_values[valid_mask == 0] = -float('inf')
            action_idx = np.argmax(q_values)
            return (action_idx // 64, action_idx % 64)
        
        action_idx = np.argmax(q_values)
        return (action_idx // self.action_size ** 0.5, action_idx % self.action_size ** 0.5)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
        
        # Learn every update_every steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.replay()
    
    def replay(self):
        """Train the Q-network on a batch of experiences."""
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1][0] * 64 + e[1][1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values (Double DQN)
        with torch.no_grad():
            # Action selection from online network
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluation from target network
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss
        loss = F.smooth_l1_loss(current_q_values, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Hard update of target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())


def train_improved_agent(n_episodes=2000, save_every=200):
    """Train the improved agent."""
    
    print("=" * 70)
    print("TRAINING IMPROVED RL AGENT")
    print("=" * 70)
    print("Key improvements:")
    print("✓ Correct random_seeder for realistic maps")
    print("✓ Better reward function emphasizing coverage")
    print("✓ Dueling DQN architecture")
    print("✓ Weighted action selection")
    print("✓ Tuned hyperparameters")
    print("-" * 70)
    
    # Setup
    env = ImprovedCirclePlacementEnv(64)
    state_size = 64 * 64 + 3  # map + radius + progress + coverage
    action_size = 64 * 64
    
    agent = ImprovedDQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=5e-4,
        epsilon_decay_steps=int(n_episodes * 0.7),
        batch_size=64,
        update_every=4
    )
    
    # Metrics
    episode_rewards = []
    episode_coverage = []
    losses = []
    
    # Training
    pbar = tqdm(range(n_episodes), desc="Training")
    
    for episode in pbar:
        # New map
        state = env.reset()
        episode_reward = 0
        
        while True:
            # Get valid actions
            valid_mask = env.get_valid_actions_mask()
            
            # Choose action
            action = agent.act(state, valid_mask)
            
            # Step
            next_state, reward, done, info = env.step(action)
            
            # Store
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update target network periodically
        if episode % 100 == 0:
            agent.update_target_network()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_coverage.append(info['coverage'])
        
        # Update progress
        pbar.set_postfix({
            'Coverage': f"{info['coverage']:.1%}",
            'Reward': f"{episode_reward:.1f}",
            'ε': f"{agent.epsilon:.3f}"
        })
        
        # Periodic evaluation
        if (episode + 1) % save_every == 0:
            print(f"\n{'='*70}")
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"{'='*70}")
            
            recent_coverage = np.mean(episode_coverage[-100:])
            best_coverage = max(episode_coverage)
            recent_rewards = np.mean(episode_rewards[-100:])
            
            print(f"Recent avg coverage: {recent_coverage:.1%}")
            print(f"Best coverage: {best_coverage:.1%}")
            print(f"Recent avg reward: {recent_rewards:.1f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            
            # Test on new maps
            print("\nTesting on 5 new maps:")
            test_coverages = []
            for _ in range(5):
                test_env = ImprovedCirclePlacementEnv(64)
                state = test_env.reset()
                
                while True:
                    valid_mask = test_env.get_valid_actions_mask()
                    with torch.no_grad():
                        action = agent.act(state, valid_mask)
                    state, _, done, info = test_env.step(action)
                    if done:
                        test_coverages.append(info['coverage'])
                        break
            
            print(f"Test coverages: {[f'{c:.1%}' for c in test_coverages]}")
            print(f"Average: {np.mean(test_coverages):.1%}")
    
    # Save model
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_coverage': episode_coverage
    }, 'improved_agent.pth')
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Final avg coverage: {np.mean(episode_coverage[-100:]):.1%}")
    print(f"Best coverage: {max(episode_coverage):.1%}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(episode_coverage, alpha=0.3)
    if len(episode_coverage) >= 100:
        ax1.plot(np.convolve(episode_coverage, np.ones(100)/100, mode='valid'), linewidth=2)
    ax1.set_ylabel('Coverage')
    ax1.set_title('Coverage Over Training')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% target')
    ax1.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='70% target')
    ax1.legend()
    
    ax2.plot(episode_rewards, alpha=0.3)
    if len(episode_rewards) >= 100:
        ax2.plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'), linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Rewards Over Training')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_training_progress.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    train_improved_agent(n_episodes=2000, save_every=200)