#!/usr/bin/env python3
"""
FAST ASYNCHRONOUS Randomized Radii Parallel Training
===================================================
High-performance asynchronous training with randomized circle configurations.
Uses the same fast async architecture as coverage_aligned_parallel_train.py.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp
import threading
import time
import gc
from collections import deque
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.algorithms.dqn_agent import GuidedDQNAgent, AdvancedCirclePlacementEnv, compute_included

@dataclass
class FastAsyncConfig:
    """Configuration for fast asynchronous training."""
    n_episodes: int = 100000
    n_workers: int = 32
    map_size: int = 64
    batch_size: int = 64
    buffer_size: int = 500000
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 20000
    target_update_freq: int = 1000
    visualize_every: int = 2000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class FastRandomizedRadiiEnvironment(AdvancedCirclePlacementEnv):
    """Fast randomized radii environment optimized for speed."""
    
    def __init__(self, map_size=64):
        # Initialize parent attributes manually first
        self.map_size = map_size
        self.current_radius_idx = 0
        self.placed_circles = []
        self.previous_coverage = 0.0
        
        # Generate random radii configuration
        self._generate_random_radii()
        
        # Now call parent's parent __init__ (skip AdvancedCirclePlacementEnv.__init__)
        super(AdvancedCirclePlacementEnv, self).__init__()
        
        # Initialize other attributes
        self.feature_extractor = None  # Skip expensive feature extraction
        self._cached_total_area = None
        self._cached_areas = None
    
    def _generate_random_radii(self):
        """Generate random radii configuration optimized for speed."""
        # Random number of circles (3-15)
        n_circles = np.random.randint(3, 16)
        
        # FAST: Generate all radii at once
        radii = np.random.randint(2, 21, size=n_circles)
        
        # Sort descending for strategic placement
        self.radii = sorted(radii, reverse=True)
        
        # Cache areas for speed
        self._cached_areas = np.array([np.pi * r * r for r in self.radii])
        self._cached_total_area = np.sum(self._cached_areas)
    
    def _calculate_max_theoretical_coverage_fast(self):
        """Fast theoretical coverage calculation."""
        if self._cached_total_area is None:
            self._cached_total_area = np.sum(self._cached_areas)
        
        map_area = self.map_size * self.map_size
        return min(1.0, self._cached_total_area / map_area)
    
    def reset(self, weighted_matrix=None):
        """Reset with new random configuration."""
        # Generate new random radii for each episode
        self._generate_random_radii()
        
        # Clear caches
        self._cached_total_area = None
        
        # Reset episode state
        self.current_radius_idx = 0
        self.placed_circles = []
        self.previous_coverage = 0.0
        
        # Generate new random map
        if weighted_matrix is None:
            # Fast random map generation
            self.original_map = np.random.exponential(scale=2.0, size=(self.map_size, self.map_size))
            self.original_map = np.clip(self.original_map, 0, 10)
        else:
            self.original_map = weighted_matrix.copy()
        
        self.current_map = self.original_map.copy()
        
        return self._get_enhanced_state()
    
    def _get_enhanced_state(self):
        """Fast state representation."""
        current_radius = self.radii[self.current_radius_idx] if self.current_radius_idx < len(self.radii) else self.radii[-1]
        
        # Fast placed mask
        placed_mask = np.zeros_like(self.current_map)
        for x, y, r in self.placed_circles:
            y_indices, x_indices = np.ogrid[:self.map_size, :self.map_size]
            mask = (x_indices - x)**2 + (y_indices - y)**2 <= r**2
            placed_mask[mask] = 1.0
        
        # Fast value density (simplified)
        value_density = self.current_map * (1 - placed_mask)
        
        # Fast features (skip expensive extraction)
        remaining_circles = len(self.radii) - self.current_radius_idx
        progress = self.current_radius_idx / len(self.radii)
        max_theoretical_coverage = self._calculate_max_theoretical_coverage_fast()
        
        # Normalized features (must be exactly 10 features to match SmartCirclePlacementNet)
        features = np.array([
            current_radius * 0.05,  # normalized radius
            remaining_circles / 15.0,  # normalized remaining
            len(self.radii) / 15.0,  # normalized total circles
            progress,  # progress
            max_theoretical_coverage,  # theoretical max
            self.current_map.sum() / (self.original_map.sum() + 1e-6),  # remaining value ratio
            self.current_map.mean() / (self.original_map.mean() + 1e-6),  # current map density
            self.current_map.max() / (self.original_map.max() + 1e-6),  # current max value ratio
            len(self.placed_circles) / max(len(self.radii), 1),  # placement progress
            1.0 if self.current_radius_idx > 0 else 0.0,  # has started placing
        ])
        
        # Verify we have exactly 10 features
        assert len(features) == 10, f"Expected 10 features, got {len(features)}"
        
        return {
            "current_map": self.current_map,
            "placed_mask": placed_mask,
            "value_density": value_density,
            "features": features,
            "current_radius": current_radius,
            "remaining_circles": remaining_circles,
            "total_circles": len(self.radii),
            "progress": progress,
            "max_theoretical_coverage": max_theoretical_coverage
        }
    
    def step(self, action):
        """Fast optimized step function."""
        x, y = action
        current_radius = self.radii[self.current_radius_idx] if self.current_radius_idx < len(self.radii) else self.radii[-1]
        
        # Calculate base collection value
        included_weight = compute_included(self.current_map, x, y, current_radius)
        
        # FAST VECTORIZED overlap penalty
        overlap_penalty = 0.0
        if self.placed_circles:
            placed_array = np.array([(px, py, pr) for px, py, pr in self.placed_circles])
            distances = np.sqrt((x - placed_array[:, 0])**2 + (y - placed_array[:, 1])**2)
            min_distances = current_radius + placed_array[:, 2]
            overlap_mask = distances < min_distances
            if np.any(overlap_mask):
                overlap_ratios = (min_distances[overlap_mask] - distances[overlap_mask]) / min_distances[overlap_mask]
                overlap_penalty = np.sum(overlap_ratios) * 2.0
        
        # Update environment state
        self.placed_circles.append((x, y, current_radius))
        
        # Fast map update
        y_indices, x_indices = np.ogrid[:self.map_size, :self.map_size]
        circle_mask = (x_indices - x)**2 + (y_indices - y)**2 <= current_radius**2
        self.current_map[circle_mask] = 0
        
        # Calculate current coverage
        current_coverage = 1.0 - (self.current_map.sum() / (self.original_map.sum() + 1e-6))
        coverage_improvement = current_coverage - self.previous_coverage
        
        # FAST REWARD CALCULATION
        reward = 0.0
        
        if included_weight > 0:
            # 1. Value collection reward
            circle_importance = (len(self.radii) - self.current_radius_idx) / len(self.radii)
            value_reward = included_weight * circle_importance * 2.0
            reward += value_reward
            
            # 2. Coverage improvement reward (normalized)
            max_theoretical_coverage = self._calculate_max_theoretical_coverage_fast()
            if max_theoretical_coverage > 0:
                coverage_reward = (coverage_improvement / max_theoretical_coverage) * 10.0
                reward += coverage_reward
            
            # 3. High-value targeting bonus
            circle_area = np.pi * current_radius * current_radius
            avg_value_in_circle = included_weight / max(circle_area, 1)
            map_avg_value = self.original_map.mean()
            if map_avg_value > 0:
                hotspot_bonus = max(0, (avg_value_in_circle / map_avg_value - 1.0) * 2.0)
                reward += hotspot_bonus
            
            # 4. Strategic placement bonus for early circles
            if self.current_radius_idx < len(self.radii) * 0.3:
                value_efficiency = included_weight / circle_area
                selectivity_bonus = value_efficiency * 1.5
                reward += selectivity_bonus
        else:
            # Heavy penalty for empty placements
            wasted_penalty = current_radius * 0.5
            reward -= wasted_penalty
        
        # Apply penalties
        reward -= overlap_penalty
        
        # Fast boundary penalty
        edge_threshold = current_radius * 1.2
        edge_distance = min(x, y, self.map_size - x, self.map_size - y)
        if edge_distance < edge_threshold:
            boundary_penalty = (edge_threshold - edge_distance) / edge_threshold
            reward -= boundary_penalty
        
        # Move to next radius
        self.current_radius_idx += 1
        done = self.current_radius_idx >= len(self.radii)
        
        # Final bonuses/penalties
        if done:
            normalized_final_coverage = current_coverage / max(self._calculate_max_theoretical_coverage_fast(), 0.1)
            
            if normalized_final_coverage > 0.9:
                reward += 15.0
            elif normalized_final_coverage > 0.8:
                reward += 10.0
            elif normalized_final_coverage > 0.7:
                reward += 5.0
            elif normalized_final_coverage > 0.6:
                reward += 2.0
            elif normalized_final_coverage < 0.3:
                reward -= 5.0
            elif normalized_final_coverage < 0.2:
                reward -= 10.0
        
        self.previous_coverage = current_coverage
        
        info = {
            "coverage": current_coverage,
            "coverage_improvement": coverage_improvement,
            "overlap_penalty": overlap_penalty,
            "radii_config": self.radii.copy(),
            "n_circles": len(self.radii),
            "max_theoretical_coverage": self._calculate_max_theoretical_coverage_fast(),
            "normalized_coverage": current_coverage / max(self._calculate_max_theoretical_coverage_fast(), 0.1)
        }
        
        return self._get_enhanced_state(), reward, done, info

class FastRandomizedHeuristicAgent:
    """Fast heuristic agent optimized for speed."""
    
    def __init__(self, map_size=64):
        self.map_size = map_size
    
    def act(self, state_dict, valid_mask=None, epsilon=0.1):
        """Fast action selection."""
        current_radius = state_dict["current_radius"]
        current_map = state_dict["current_map"]
        progress = state_dict.get("progress", 0.5)
        
        if np.random.random() < epsilon:
            # FAST EXPLORATION
            if valid_mask is not None:
                valid_positions = np.argwhere(valid_mask.reshape(self.map_size, self.map_size))
                if len(valid_positions) > 0:
                    idx = np.random.randint(len(valid_positions))
                    return tuple(valid_positions[idx])
            
            # Random valid position
            return (
                np.random.randint(current_radius, self.map_size - current_radius),
                np.random.randint(current_radius, self.map_size - current_radius)
            )
        else:
            # FAST GREEDY: Reduced samples for speed
            n_samples = 15  # Much fewer samples
            
            # Fast hotspot detection
            hotspot_threshold = np.percentile(current_map, 85)
            hotspots = np.argwhere(current_map > hotspot_threshold)
            
            # Limit hotspots
            if len(hotspots) > 30:
                hotspot_indices = np.random.choice(len(hotspots), 30, replace=False)
                hotspots = hotspots[hotspot_indices]
            
            best_score = -float('inf')
            best_action = None
            
            # Fast candidate generation
            candidates = []
            if len(hotspots) > 0:
                # Sample around hotspots
                n_hotspot = min(10, len(hotspots))
                selected_hotspots = hotspots[np.random.choice(len(hotspots), n_hotspot, replace=True)]
                
                for hx, hy in selected_hotspots:
                    search_radius = min(current_radius, 10)
                    x = np.clip(hx + np.random.randint(-search_radius, search_radius + 1), 
                               current_radius, self.map_size - current_radius - 1)
                    y = np.clip(hy + np.random.randint(-search_radius, search_radius + 1), 
                               current_radius, self.map_size - current_radius - 1)
                    candidates.append((x, y))
            
            # Add some random candidates
            for _ in range(5):
                x = np.random.randint(current_radius, self.map_size - current_radius)
                y = np.random.randint(current_radius, self.map_size - current_radius)
                candidates.append((x, y))
            
            # Fast evaluation
            for x, y in candidates:
                # Fast local value calculation
                local_value = compute_included(current_map, x, y, current_radius)
                
                if local_value > 0:
                    score = local_value
                    
                    # Fast overlap check (simplified)
                    if state_dict.get("placed_circles", []):
                        min_dist = float('inf')
                        for px, py, pr in state_dict.get("placed_circles", []):
                            dist = np.sqrt((x - px)**2 + (y - py)**2)
                            min_dist = min(min_dist, dist - pr - current_radius)
                        
                        if min_dist < 0:  # Overlap
                            score *= 0.1  # Heavy penalty
                    
                    # Fast boundary penalty
                    edge_distance = min(x, y, self.map_size - x, self.map_size - y)
                    if edge_distance < current_radius * 1.2:
                        score *= 0.8
                    
                    if score > best_score:
                        best_score = score
                        best_action = (x, y)
            
            if best_action is not None:
                return best_action
            
            # Fallback
            return (
                np.random.randint(current_radius, self.map_size - current_radius),
                np.random.randint(current_radius, self.map_size - current_radius)
            )

def fast_async_randomized_worker_process(config, result_queue, epsilon_value, worker_id):
    """Fast asynchronous worker process."""
    np.random.seed(worker_id * 1000 + int(time.time()) % 1000)
    
    env = FastRandomizedRadiiEnvironment(map_size=config.map_size)
    agent = FastRandomizedHeuristicAgent(map_size=config.map_size)
    
    episodes_completed = 0
    radii_configs_generated = 0
    
    while True:
        try:
            # Get current epsilon
            with epsilon_value.get_lock():
                current_epsilon = epsilon_value.value
            
            # Run episode
            state = env.reset()
            episode_reward = 0
            episode_experiences = []
            total_coverage_improvement = 0
            
            radii_config = env.radii.copy()
            n_circles = len(radii_config)
            max_theoretical_coverage = env._calculate_max_theoretical_coverage_fast()
            radii_configs_generated += 1
            
            while True:
                # Get valid actions mask
                current_radius = state["current_radius"]
                valid_mask = np.ones((config.map_size, config.map_size), dtype=bool)
                valid_mask[:current_radius, :] = False
                valid_mask[-current_radius:, :] = False
                valid_mask[:, :current_radius] = False
                valid_mask[:, -current_radius:] = False
                
                # Add placed circles to state for heuristic agent
                state_with_circles = state.copy()
                state_with_circles["placed_circles"] = env.placed_circles
                
                # Get action from heuristic agent
                action = agent.act(state_with_circles, valid_mask.flatten(), current_epsilon)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                experience = (state, action, reward, next_state if not done else None, done)
                episode_experiences.append(experience)
                
                episode_reward += reward
                total_coverage_improvement += info.get("coverage_improvement", 0)
                
                if done:
                    break
                
                state = next_state
            
            episodes_completed += 1
            
            # Send results
            result_data = {
                "experiences": episode_experiences,
                "episode_reward": episode_reward,
                "coverage": info["coverage"],
                "total_coverage_improvement": total_coverage_improvement,
                "avg_coverage_improvement": total_coverage_improvement / max(len(episode_experiences), 1),
                "worker_id": worker_id,
                "episodes_completed": episodes_completed,
                "radii_configs_generated": radii_configs_generated,
                "radii_config": radii_config,
                "n_circles": n_circles,
                "max_theoretical_coverage": max_theoretical_coverage,
                "normalized_coverage": info.get("normalized_coverage", 0),
                "map_diversity_stats": {
                    "map_mean": env.original_map.mean(),
                    "map_std": env.original_map.std(),
                    "map_max": env.original_map.max()
                }
            }
            
            result_queue.put(result_data)
            
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            continue

class ThreadSafeReplayBuffer:
    """Thread-safe replay buffer."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
    
    def push(self, experience):
        with self.lock:
            self.buffer.append(experience)
    
    def sample(self, batch_size):
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def __len__(self):
        with self.lock:
            return len(self.buffer)

class FastAsyncRandomizedRadiiTrainer:
    """Fast asynchronous randomized radii trainer."""
    
    def __init__(self, config: FastAsyncConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize agent
        self.agent = GuidedDQNAgent(
            map_size=config.map_size,
            learning_rate=config.learning_rate
        )
        
        # Ensure agent uses the correct device
        self.device = self.agent.device
        
        # Mixed precision training
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        
        # Shared epsilon value
        self.epsilon_value = mp.Value('f', config.epsilon_start)
        
        # Result queue for async communication
        self.result_queue = mp.Queue(maxsize=config.n_workers * 2)
        
        # Replay buffer
        self.replay_buffer = ThreadSafeReplayBuffer(config.buffer_size)
        
        # Start worker processes
        self.workers = []
        for i in range(config.n_workers):
            worker = mp.Process(
                target=fast_async_randomized_worker_process,
                args=(config, self.result_queue, self.epsilon_value, i),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_coverage = []
        self.normalized_coverage = []
        self.coverage_improvements = []
        self.avg_coverage_improvements = []
        self.losses = []
        self.training_step = 0
        
        # Randomized radii metrics
        self.radii_configs = []
        self.n_circles_history = []
        self.max_theoretical_coverage_history = []
        
        # Map diversity tracking
        self.map_diversity_stats = []
        self.worker_stats = {}
        
        # Experience collection thread
        self.collection_thread = threading.Thread(target=self._collect_experiences, daemon=True)
        self.collection_thread.start()
        
        # Debug flag
        self._debug_shapes = False
    
    def _collect_experiences(self):
        """Asynchronously collect experiences from workers."""
        while True:
            try:
                result_data = self.result_queue.get(timeout=1.0)
                
                # Add experiences to buffer
                for exp in result_data["experiences"]:
                    self.replay_buffer.push(exp)
                
                # Record metrics
                self.episode_rewards.append(result_data["episode_reward"])
                self.episode_coverage.append(result_data["coverage"])
                self.normalized_coverage.append(result_data.get("normalized_coverage", 0))
                self.coverage_improvements.append(result_data["total_coverage_improvement"])
                self.avg_coverage_improvements.append(result_data["avg_coverage_improvement"])
                
                # Record randomized radii metrics
                self.radii_configs.append(result_data["radii_config"])
                self.n_circles_history.append(result_data["n_circles"])
                self.max_theoretical_coverage_history.append(result_data["max_theoretical_coverage"])
                
                # Track map diversity
                if "map_diversity_stats" in result_data:
                    self.map_diversity_stats.append(result_data["map_diversity_stats"])
                
                # Update worker statistics
                worker_id = result_data["worker_id"]
                self.worker_stats[worker_id] = {
                    "episodes_completed": result_data.get("episodes_completed", 0),
                    "radii_configs_generated": result_data.get("radii_configs_generated", 0)
                }
                
            except:
                continue
    
    def _update_epsilon(self):
        """Update shared epsilon value."""
        current_episodes = len(self.episode_rewards)
        
        if current_episodes < self.config.epsilon_decay_episodes:
            decay_progress = current_episodes / self.config.epsilon_decay_episodes
            new_epsilon = self.config.epsilon_start - (self.config.epsilon_start - self.config.epsilon_end) * decay_progress
        else:
            new_epsilon = self.config.epsilon_end
        
        with self.epsilon_value.get_lock():
            self.epsilon_value.value = new_epsilon
            self.agent.epsilon = new_epsilon
    
    def _prepare_batch_tensors_optimized(self, batch):
        """Optimized tensor preparation."""
        states = [e[0] for e in batch]
        actions = [e[1] for e in batch]
        rewards = [e[2] for e in batch]
        next_states = [e[3] for e in batch if e[3] is not None]
        dones = [e[4] for e in batch]
        
        # Convert to numpy arrays first, then to tensors
        current_maps_np = np.stack([s["current_map"] for s in states])
        placed_masks_np = np.stack([s["placed_mask"] for s in states])
        value_densities_np = np.stack([s["value_density"] for s in states])
        features_np = np.stack([s["features"] for s in states])
        
        # Convert to tensors
        current_maps = torch.from_numpy(current_maps_np).float().to(self.device)
        placed_masks = torch.from_numpy(placed_masks_np).float().to(self.device)
        value_densities = torch.from_numpy(value_densities_np).float().to(self.device)
        features = torch.from_numpy(features_np).float().to(self.device)
        
        state_batch = {
            "current_map": current_maps,
            "placed_mask": placed_masks,
            "value_density": value_densities,
            "features": features,
        }
        
        return state_batch, actions, rewards, next_states, dones
    
    def _train_step(self):
        """Optimized training step with mixed precision."""
        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return None
        
        try:
            state_batch, actions, rewards, next_states, dones = self._prepare_batch_tensors_optimized(batch)
            
            # Forward pass
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    current_q_values = self.agent.q_network(state_batch)
                    
                    # Debug: Check tensor shapes
                    if hasattr(self, '_debug_shapes') and not self._debug_shapes:
                        print(f"Debug - Batch shapes:")
                        print(f"  current_map: {state_batch['current_map'].shape}")
                        print(f"  placed_mask: {state_batch['placed_mask'].shape}")
                        print(f"  value_density: {state_batch['value_density'].shape}")
                        print(f"  features: {state_batch['features'].shape}")
                        print(f"  q_values shape: {current_q_values.shape}")
                        print(f"  action_indices length: {len(action_indices)}")
                        self._debug_shapes = True
                    
                    # Convert actions to tensor indices
                    action_indices = []
                    for action in actions:
                        if isinstance(action, (list, tuple)) and len(action) == 2:
                            x, y = action
                            idx = x * self.config.map_size + y
                        else:
                            idx = action
                        action_indices.append(idx)
                    
                    action_indices = torch.LongTensor(action_indices).to(self.device)
                    # Fix dimension mismatch: ensure action_indices has correct shape
                    if current_q_values.dim() == 3:  # Shape: [batch, height, width]
                        current_q_values = current_q_values.view(current_q_values.size(0), -1)  # Flatten to [batch, height*width]
                    current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
                    
                    # Calculate target Q-values
                    target_q_values = torch.FloatTensor(rewards).to(self.device)
                    
                    # Calculate loss
                    loss = nn.MSELoss()(current_q_values, target_q_values)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.agent.optimizer)
                self.scaler.update()
            else:
                # CPU training
                current_q_values = self.agent.q_network(state_batch)
                
                # Debug: Check tensor shapes
                if hasattr(self, '_debug_shapes') and not self._debug_shapes:
                    print(f"Debug - Batch shapes (CPU):")
                    print(f"  current_map: {state_batch['current_map'].shape}")
                    print(f"  placed_mask: {state_batch['placed_mask'].shape}")
                    print(f"  value_density: {state_batch['value_density'].shape}")
                    print(f"  features: {state_batch['features'].shape}")
                    print(f"  q_values shape: {current_q_values.shape}")
                    self._debug_shapes = True
                
                action_indices = []
                for action in actions:
                    if isinstance(action, (list, tuple)) and len(action) == 2:
                        x, y = action
                        idx = x * self.config.map_size + y
                    else:
                        idx = action
                    action_indices.append(idx)
                
                action_indices = torch.LongTensor(action_indices).to(self.device)
                # Fix dimension mismatch: ensure action_indices has correct shape
                if current_q_values.dim() == 3:  # Shape: [batch, height, width]
                    current_q_values = current_q_values.view(current_q_values.size(0), -1)  # Flatten to [batch, height*width]
                current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
                
                target_q_values = torch.FloatTensor(rewards).to(self.device)
                loss = nn.MSELoss()(current_q_values, target_q_values)
                
                self.agent.optimizer.zero_grad()
                loss.backward()
                self.agent.optimizer.step()
            
            self.training_step += 1
            
            # Update target network
            if self.training_step % self.config.target_update_freq == 0:
                self.agent.update_target_network()
            
            return loss.item()
            
        except Exception as e:
            print(f"Training step error: {e}")
            return None
    
    def visualize_strategy(self, episode: int, save_path: str = None):
        """Visualize current strategy with randomized radii."""
        if save_path is None:
            save_path = f"randomized_radii_strategy_episode_{episode}.png"
        
        # Create test environment
        test_env = FastRandomizedRadiiEnvironment(map_size=self.config.map_size)
        state = test_env.reset()
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Configuration info
        radii_config = test_env.radii
        n_circles = len(radii_config)
        max_theoretical = test_env._calculate_max_theoretical_coverage_fast()
        
        # Test episode
        coverage_history = [0]
        step_idx = 0
        steps_to_show = [1, 3, 5]  # Show specific steps
        
        for i in range(n_circles):
            current_radius = state["current_radius"]
            
            # Get valid actions
            valid_mask = np.ones((self.config.map_size, self.config.map_size), dtype=bool)
            valid_mask[:current_radius, :] = False
            valid_mask[-current_radius:, :] = False
            valid_mask[:, :current_radius] = False
            valid_mask[:, -current_radius:] = False
            
            # Get agent's action
            with torch.no_grad():
                state_batch = self.agent._prepare_state_batch([state])
                q_values = self.agent.q_network(state_batch).squeeze(0)
                
                if valid_mask is not None:
                    # Fix mask application: ensure shapes match
                    if q_values.dim() == 2:  # Shape: [height, width]
                        mask_tensor = torch.FloatTensor(valid_mask).to(self.device)
                    else:  # Flattened
                        mask_tensor = torch.FloatTensor(valid_mask.flatten()).to(self.device)
                    
                    # Apply mask
                    q_values_flat = q_values.view(-1)
                    mask_flat = mask_tensor.view(-1)
                    q_values_flat = q_values_flat + (mask_flat - 1) * 1e10
                    
                    action_idx = q_values_flat.argmax().item()
                else:
                    action_idx = q_values.view(-1).argmax().item()
                
                action = (action_idx // self.config.map_size, action_idx % self.config.map_size)
            
            # Take step
            state, reward, done, info = test_env.step(action)
            coverage_history.append(info["coverage"])
            
            # Visualize at specific steps
            if i + 1 in steps_to_show and step_idx < 3:
                ax = axes[step_idx]
                ax.imshow(test_env.original_map, cmap="hot", alpha=0.6)
                
                # Draw circles with color coding by size
                colors = plt.cm.viridis(np.linspace(0, 1, len(radii_config)))
                for j, (x, y, r) in enumerate(test_env.placed_circles):
                    color_idx = radii_config.index(r) if r in radii_config else 0
                    circle = plt.Circle((y, x), r, fill=False, color=colors[color_idx], linewidth=2)
                    ax.add_patch(circle)
                
                coverage_improvement = coverage_history[-1] - coverage_history[-2] if len(coverage_history) > 1 else 0
                ax.set_title(f"Step {i + 1}: Coverage {info['coverage']:.1%} (+{coverage_improvement:.1%})\nRadius: {current_radius}")
                ax.axis("off")
                
                step_idx += 1
            
            if done:
                break
        
        # Final result
        axes[3].imshow(test_env.original_map, cmap="hot", alpha=0.6)
        colors = plt.cm.viridis(np.linspace(0, 1, len(radii_config)))
        for j, (x, y, r) in enumerate(test_env.placed_circles):
            color_idx = radii_config.index(r) if r in radii_config else 0
            circle = plt.Circle((y, x), r, fill=False, color=colors[color_idx], linewidth=2)
            axes[3].add_patch(circle)
        
        final_coverage = coverage_history[-1]
        normalized_coverage = final_coverage / max(max_theoretical, 0.1)
        axes[3].set_title(f"Final: {final_coverage:.1%} (Norm: {normalized_coverage:.1%})")
        axes[3].axis("off")
        
        # Coverage progress
        axes[4].plot(coverage_history, 'g-', linewidth=2, marker='o')
        axes[4].set_title("Coverage Progress")
        axes[4].set_xlabel("Circle Placement")
        axes[4].set_ylabel("Coverage")
        axes[4].grid(True)
        
        # Configuration and stats
        config_text = f"Episode: {episode}\n"
        config_text += f"Radii Config: {radii_config}\n"
        config_text += f"Circles: {n_circles}\n"
        config_text += f"Theoretical Max: {max_theoretical:.1%}\n"
        config_text += f"Training Steps: {self.training_step:,}\n"
        config_text += f"Buffer Size: {len(self.replay_buffer):,}"
        
        if len(self.episode_rewards) > 100:
            recent_rewards = self.episode_rewards[-100:]
            recent_coverage = self.normalized_coverage[-100:]
            correlation = np.corrcoef(recent_rewards, recent_coverage)[0, 1]
            config_text += f"\nReward-NormCov Corr: {correlation:.3f}"
        
        axes[5].text(0.1, 0.9, config_text, fontsize=10, transform=axes[5].transAxes, verticalalignment='top')
        axes[5].set_title("Configuration & Stats")
        axes[5].axis("off")
        
        plt.suptitle(f"Fast Async Randomized Radii Agent - Episode {episode}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _quick_visualize(self, episode: int):
        """Quick visualization every 100 episodes."""
        if episode % 100 == 0:
            save_path = f"fast_async_randomized_episode_{episode}.png"
            self.visualize_strategy(episode, save_path)
    
    def train(self):
        """Fast asynchronous training loop."""
        print("=" * 100)
        print(f"FAST ASYNC RANDOMIZED RADII TRAINING WITH {self.config.n_workers} WORKERS")
        print("=" * 100)
        print(f"Target episodes: {self.config.n_episodes:,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Buffer size: {self.config.buffer_size:,}")
        print(f"Workers: {self.config.n_workers}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Randomized radii: 3-15 circles, radii 2-20")
        print("=" * 100)
        
        pbar = tqdm(total=self.config.n_episodes, desc="Fast Async Randomized Training")
        last_episode_count = 0
        start_time = time.time()
        
        while len(self.episode_rewards) < self.config.n_episodes:
            current_episodes = len(self.episode_rewards)
            pbar.update(current_episodes - last_episode_count)
            last_episode_count = current_episodes
            
            # Update epsilon
            self._update_epsilon()
            
            # Train if we have enough experiences
            if len(self.replay_buffer) > self.config.batch_size * self.config.gradient_accumulation_steps:
                total_loss = 0
                num_batches = 0
                
                for _ in range(self.config.gradient_accumulation_steps):
                    loss = self._train_step()
                    if loss is not None:
                        total_loss += loss
                        num_batches += 1
                
                if num_batches > 0:
                    self.losses.append(total_loss / num_batches)
            
            # Update progress bar
            if current_episodes > 0:
                recent_coverage = np.mean(self.episode_coverage[-1000:]) if len(self.episode_coverage) >= 1000 else np.mean(self.episode_coverage) if self.episode_coverage else 0
                recent_norm_cov = np.mean(self.normalized_coverage[-1000:]) if len(self.normalized_coverage) >= 1000 else np.mean(self.normalized_coverage) if self.normalized_coverage else 0
                recent_reward = np.mean(self.episode_rewards[-1000:]) if len(self.episode_rewards) >= 1000 else np.mean(self.episode_rewards) if self.episode_rewards else 0
                recent_circles = np.mean(self.n_circles_history[-1000:]) if len(self.n_circles_history) >= 1000 else np.mean(self.n_circles_history) if self.n_circles_history else 0
                
                with self.epsilon_value.get_lock():
                    current_epsilon = self.epsilon_value.value
                
                pbar.set_postfix({
                    "Coverage": f"{recent_coverage:.1%}",
                    "NormCov": f"{recent_norm_cov:.1%}",
                    "Reward": f"{recent_reward:.2f}",
                    "AvgCircles": f"{recent_circles:.1f}",
                    "Buffer": f"{len(self.replay_buffer):,}",
                    "Loss": f"{np.mean(self.losses[-100:]):.4f}" if self.losses else "N/A",
                    "Epsilon": f"{current_epsilon:.3f}"
                })
            
            # Quick visualization (temporarily disabled for debugging)
            # if current_episodes > 0 and current_episodes > 2000:  # Only after some training
            #     self._quick_visualize(current_episodes)
            
            # Periodic evaluation (temporarily disabled for debugging)
            # if current_episodes > 0 and current_episodes % self.config.visualize_every == 0:
            #     self._evaluate_and_visualize(current_episodes)
            
            # Progress updates
            if current_episodes > 0 and current_episodes % 200 == 0:
                self._print_progress_update(current_episodes)
            
            # Periodic cleanup
            if current_episodes % 500 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            time.sleep(0.01)  # Prevent busy waiting
        
        pbar.close()
        
        # Final statistics
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n" + "=" * 100)
        print("FAST ASYNC RANDOMIZED RADII TRAINING COMPLETE!")
        print("=" * 100)
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        print(f"Episodes per second: {self.config.n_episodes / training_time:.2f}")
        print(f"Final average coverage: {np.mean(self.episode_coverage[-1000:]):.1%}")
        print(f"Final average normalized coverage: {np.mean(self.normalized_coverage[-1000:]):.1%}")
        print(f"Best coverage achieved: {max(self.episode_coverage):.1%}")
        print(f"Average circles per episode: {np.mean(self.n_circles_history):.1f}")
        print(f"Total training steps: {self.training_step:,}")
        
        self._save_final_model()
        self._cleanup()
    
    def _print_progress_update(self, episode: int):
        """Print detailed progress update."""
        if len(self.episode_rewards) < 50:
            return
        
        recent_coverage = np.mean(self.episode_coverage[-200:]) if len(self.episode_coverage) >= 200 else np.mean(self.episode_coverage)
        recent_norm_cov = np.mean(self.normalized_coverage[-200:]) if len(self.normalized_coverage) >= 200 else np.mean(self.normalized_coverage)
        recent_rewards = np.mean(self.episode_rewards[-200:]) if len(self.episode_rewards) >= 200 else np.mean(self.episode_rewards)
        best_coverage = max(self.episode_coverage)
        best_norm_cov = max(self.normalized_coverage)
        
        # Configuration diversity stats
        recent_circles = self.n_circles_history[-200:] if len(self.n_circles_history) >= 200 else self.n_circles_history
        avg_circles = np.mean(recent_circles)
        circles_range = f"{min(recent_circles)}-{max(recent_circles)}"
        
        recent_theoretical = self.max_theoretical_coverage_history[-200:] if len(self.max_theoretical_coverage_history) >= 200 else self.max_theoretical_coverage_history
        avg_theoretical = np.mean(recent_theoretical)
        
        # Correlation
        correlation = 0.0
        if len(self.episode_rewards) > 100:
            correlation = np.corrcoef(self.episode_rewards[-500:], self.normalized_coverage[-500:])[0, 1]
        
        with self.epsilon_value.get_lock():
            current_epsilon = self.epsilon_value.value
        
        print(f"\n🚀 Episode {episode:,} Progress (Fast Async Randomized Radii):")
        print(f"   Coverage: {recent_coverage:.1%} (Best: {best_coverage:.1%})")
        print(f"   Normalized Coverage: {recent_norm_cov:.1%} (Best: {best_norm_cov:.1%})")
        print(f"   Reward: {recent_rewards:.2f}")
        print(f"   Reward-NormCov Correlation: {correlation:.3f}")
        print(f"   Epsilon: {current_epsilon:.3f}")
        print(f"   Training Steps: {self.training_step:,}")
        print(f"   Buffer: {len(self.replay_buffer):,}/{self.config.buffer_size:,}")
        print(f"   Configuration Diversity:")
        print(f"     • Avg Circles: {avg_circles:.1f} (Range: {circles_range})")
        print(f"     • Avg Theoretical Max: {avg_theoretical:.1%}")
        
        # Recent trends
        if len(self.episode_rewards) > 10:
            recent_10_rewards = self.episode_rewards[-10:]
            recent_10_norm_cov = self.normalized_coverage[-10:]
            print(f"   📈 Recent trend: Reward {np.mean(recent_10_rewards[-5:]):.2f} vs {np.mean(recent_10_rewards[:5]):.2f}")
            print(f"   📈 Recent trend: NormCov {np.mean(recent_10_norm_cov[-5:]):.1%} vs {np.mean(recent_10_norm_cov[:5]):.1%}")
        
        # Worker stats
        if self.worker_stats:
            total_episodes = sum(stats.get("episodes_completed", 0) for stats in self.worker_stats.values())
            total_configs = sum(stats.get("radii_configs_generated", 0) for stats in self.worker_stats.values())
            print(f"   Workers: {len(self.worker_stats)} active, {total_episodes:,} episodes, {total_configs:,} configs generated")
    
    def _evaluate_and_visualize(self, episode: int):
        """Evaluate and create detailed visualization."""
        save_path = f"fast_async_randomized_evaluation_episode_{episode}.png"
        self.visualize_strategy(episode, save_path)
        print(f"   💾 Saved evaluation visualization: {save_path}")
    
    def _save_final_model(self):
        """Save the trained model and metrics."""
        model_path = "fast_async_randomized_radii_dqn_model.pth"
        
        save_data = {
            "model_state_dict": self.agent.q_network.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "training_step": self.training_step,
            "episode_rewards": self.episode_rewards,
            "episode_coverage": self.episode_coverage,
            "normalized_coverage": self.normalized_coverage,
            "n_circles_history": self.n_circles_history,
            "max_theoretical_coverage_history": self.max_theoretical_coverage_history,
            "config": self.config
        }
        
        torch.save(save_data, model_path)
        print(f"   💾 Model saved: {model_path}")
    
    def _cleanup(self):
        """Clean up resources."""
        for worker in self.workers:
            worker.terminate()
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

def main():
    """Main training function."""
    # Detect system capabilities
    n_cores = mp.cpu_count()
    n_workers = min(n_cores, 64)  # Cap at 64 workers
    
    print(f"System: {n_cores} cores")
    print(f"Using {n_workers} workers for fast async randomized radii training")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Configuration
    config = FastAsyncConfig(
        n_episodes=100000,
        n_workers=n_workers,
        map_size=64,
        batch_size=64,
        buffer_size=500000,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_episodes=20000,
        target_update_freq=1000,
        visualize_every=2000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create trainer and start training
    trainer = FastAsyncRandomizedRadiiTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()