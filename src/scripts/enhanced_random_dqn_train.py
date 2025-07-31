#!/usr/bin/env python3
"""
Enhanced Randomized Radii Parallel Training
==========================================
Small improvements over random_dqn_train.py for better performance.
"""

import sys
import os
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.algorithms.dqn_agent import (
    AdvancedCirclePlacementEnv,
    GuidedDQNAgent,
    random_seeder,
    compute_included,
    HeatmapFeatureExtractor,
)
from src.utils.periodic_tracker import PeriodicTaskTracker, RobustPeriodicChecker


class EnhancedRandomizedRadiiEnvironment(AdvancedCirclePlacementEnv):
    """Enhanced environment with better reward shaping and features."""
    
    def __init__(self, map_size=128):
        self.previous_coverage = 0.0
        # Generate initial random radii - will be regenerated on each reset
        self._generate_random_radii()
        super().__init__(map_size, radii=self.radii)
        
    def _generate_random_radii(self):
        """Generate random radii configuration with better distribution."""
        # Random number of circles (3-15) with slight bias towards more circles
        n_circles = np.random.choice(range(3, 16), p=self._get_circle_distribution())
        
        # Generate random radii (2-20) with better size distribution
        radii = []
        for _ in range(n_circles):
            # Bias towards medium-sized circles for better packing
            if np.random.random() < 0.2:  # 20% large circles
                radius = np.random.randint(15, 21)
            elif np.random.random() < 0.5:  # 40% medium circles
                radius = np.random.randint(8, 15)
            else:  # 40% small circles
                radius = np.random.randint(2, 8)
            radii.append(radius)
        
        # Sort descending for strategic placement
        self.radii = sorted(radii, reverse=True)
    
    def _get_circle_distribution(self):
        """Get probability distribution for number of circles."""
        # Slight bias towards 7-12 circles (sweet spot for coverage)
        probs = np.ones(13)  # 3 to 15 circles
        probs[4:10] *= 1.5  # Boost probability for 7-12 circles
        return probs / probs.sum()
        
    def reset(self, weighted_matrix=None):
        """Reset environment with new random radii configuration."""
        # Generate new random radii for each episode
        self._generate_random_radii()
        
        if weighted_matrix is None:
            # Generate new random map using proper random_seeder
            weighted_matrix = random_seeder(self.map_size, time_steps=100000)
        
        # Reset environment state manually (avoid recursion)
        self.current_radius_idx = 0
        self.placed_circles = []
        self.placement_order = []
        self.total_weight_collected = 0
        self.coverage_history = []
        
        # Set up maps
        self.original_map = weighted_matrix.copy()
        self.current_map = weighted_matrix.copy()
        
        # Initialize feature extractor if needed
        if not hasattr(self, 'feature_extractor'):
            self.feature_extractor = HeatmapFeatureExtractor(self.map_size)
        
        self.previous_coverage = 0.0
        return self._get_enhanced_state()
    
    def step(self, action):
        """Enhanced step with better reward function."""
        x, y = action
        radius = self.radii[self.current_radius_idx]
        
        # Store current coverage before action
        current_coverage = 1 - (self.current_map.sum() / (self.original_map.sum() + 1e-8))
        
        # Calculate base collection value
        included_weight = compute_included(self.current_map, x, y, radius)
        
        # Enhanced overlap detection with gradual penalty
        overlap_penalty = 0.0
        touching_bonus = 0.0
        for px, py, pr in self.placed_circles:
            distance = np.sqrt((x - px)**2 + (y - py)**2)
            min_distance = radius + pr
            
            if distance < min_distance:
                # Overlap detected - gradual penalty
                overlap_ratio = (min_distance - distance) / min_distance
                overlap_penalty += overlap_ratio * 5.0
            elif distance < min_distance + 2:  # Nearly touching (within 2 pixels)
                # Bonus for tight packing
                touching_bonus += 0.5
        
        # Update environment state
        self.placed_circles.append((x, y, radius))
        self.placement_order.append(radius)
        self.total_weight_collected += included_weight
        
        # Update current map
        for i in range(
            max(0, int(x - radius)), min(self.map_size, int(x + radius + 1))
        ):
            for j in range(
                max(0, int(y - radius)), min(self.map_size, int(y + radius + 1))
            ):
                if (i - x) ** 2 + (j - y) ** 2 <= radius**2:
                    self.current_map[i, j] = 0
        
        # Calculate new coverage after action
        new_coverage = 1 - (self.current_map.sum() / (self.original_map.sum() + 1e-8))
        
        # ENHANCED REWARD FUNCTION
        reward = 0.0
        
        # 1. Basic reward: value collected (scaled)
        reward += included_weight * 0.1
        
        # 2. Coverage improvement bonus
        coverage_improvement = new_coverage - current_coverage
        if coverage_improvement > 0:
            reward += coverage_improvement * 15.0  # Increased from 10.0
        
        # 3. Overlap penalty
        reward -= overlap_penalty
        
        # 4. Tight packing bonus
        reward += touching_bonus
        
        # 5. Efficiency bonus for good value/area ratio
        circle_area = np.pi * radius * radius
        efficiency = 0
        if circle_area > 0 and included_weight > 0:
            efficiency = included_weight / circle_area
            avg_map_value = self.original_map.mean()
            if efficiency > avg_map_value:
                reward += 1.0  # Bonus for efficient placement
        
        # 6. Progressive bonus for later circles (encourage completion)
        progress_ratio = self.current_radius_idx / len(self.radii)
        if progress_ratio > 0.5:
            reward += progress_ratio * 0.5
        
        # 7. Boundary penalty (reduced)
        edge_distance = min(x, y, self.map_size - x, self.map_size - y)
        if edge_distance < radius:
            boundary_penalty = (radius - edge_distance) / radius
            reward -= boundary_penalty * 1.0  # Reduced from 2.0
        
        # Move to next radius
        self.current_radius_idx += 1
        done = self.current_radius_idx >= len(self.radii)
        
        # Final bonus for good coverage (adjusted thresholds)
        if done:
            if new_coverage > 0.7:  # Lowered from 0.8
                reward += 15.0
            elif new_coverage > 0.5:  # Lowered from 0.6
                reward += 8.0
            elif new_coverage > 0.35:  # Lowered from 0.4
                reward += 4.0
            elif new_coverage < 0.2:
                reward -= 5.0
        
        # Update coverage tracking
        self.coverage_history.append(new_coverage)
        self.previous_coverage = new_coverage
        
        info = {
            "coverage": new_coverage,
            "coverage_improvement": coverage_improvement,
            "weight_collected": included_weight,
            "total_weight": self.total_weight_collected,
            "radii_config": self.radii.copy(),
            "n_circles": len(self.radii),
            "overlap_penalty": overlap_penalty,
            "touching_bonus": touching_bonus,
            "efficiency": efficiency,
        }
        
        return self._get_enhanced_state(), reward, done, info


class EnhancedHeuristicAgent:
    """Enhanced heuristic agent with better strategies."""
    
    def __init__(self, map_size=128):
        self.map_size = map_size
    
    def act(self, state_dict, valid_mask=None, epsilon=0.1):
        """Enhanced action selection with smarter heuristics."""
        current_radius = state_dict.get("current_radius", 5)
        current_map = state_dict["current_map"]
        placed_circles = state_dict.get("placed_circles", [])
        
        if np.random.random() < epsilon:
            # Smart exploration - bias towards unexplored high-value areas
            if valid_mask is not None:
                # Weight valid positions by local value
                valid_positions = np.argwhere(valid_mask.reshape(self.map_size, self.map_size))
                if len(valid_positions) > 0:
                    # Calculate local values for valid positions
                    values = []
                    sample_size = min(100, len(valid_positions))
                    sampled_positions = valid_positions[np.random.choice(len(valid_positions), sample_size, replace=False)]
                    
                    for pos in sampled_positions:
                        local_value = self._calculate_local_value(current_map, pos[0], pos[1], current_radius)
                        values.append(local_value)
                    
                    if values and max(values) > 0:
                        # Weighted random selection based on value
                        values = np.array(values)
                        probs = values / values.sum()
                        idx = np.random.choice(len(values), p=probs)
                        return tuple(sampled_positions[idx])
            
            # Fallback to random
            return (
                np.random.randint(current_radius, self.map_size - current_radius),
                np.random.randint(current_radius, self.map_size - current_radius)
            )
        else:
            # Enhanced greedy strategy
            best_value = -1
            best_action = None
            
            # Strategy 1: Try positions near existing circles for tight packing
            if placed_circles and np.random.random() < 0.3:
                for px, py, pr in placed_circles[-3:]:  # Check last 3 circles
                    # Try positions around this circle
                    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                    for angle in angles:
                        # Position at touching distance
                        dist = pr + current_radius + 1
                        x = int(px + dist * np.cos(angle))
                        y = int(py + dist * np.sin(angle))
                        
                        # Check bounds
                        if (current_radius <= x < self.map_size - current_radius and 
                            current_radius <= y < self.map_size - current_radius):
                            local_value = self._calculate_local_value(current_map, x, y, current_radius)
                            if local_value > best_value:
                                best_value = local_value
                                best_action = (x, y)
            
            # Strategy 2: Find high-value clusters
            high_value_threshold = np.percentile(current_map[current_map > 0], 75) if current_map.max() > 0 else 0
            high_value_positions = np.argwhere(current_map > high_value_threshold)
            
            if len(high_value_positions) > 0:
                # Sample positions around high-value areas
                n_samples = min(20, len(high_value_positions))
                sampled_positions = high_value_positions[np.random.choice(len(high_value_positions), n_samples, replace=False)]
                
                for center in sampled_positions:
                    # Try the center and nearby positions
                    for dx in [-current_radius//2, 0, current_radius//2]:
                        for dy in [-current_radius//2, 0, current_radius//2]:
                            x = np.clip(center[0] + dx, current_radius, self.map_size - current_radius - 1)
                            y = np.clip(center[1] + dy, current_radius, self.map_size - current_radius - 1)
                            
                            local_value = self._calculate_local_value(current_map, x, y, current_radius)
                            if local_value > best_value:
                                best_value = local_value
                                best_action = (x, y)
            
            if best_action is not None:
                return best_action
            
            # Fallback
            return (
                np.random.randint(current_radius, self.map_size - current_radius),
                np.random.randint(current_radius, self.map_size - current_radius)
            )
    
    def _calculate_local_value(self, current_map, x, y, radius):
        """Calculate the value a circle would collect at position (x, y)."""
        value = 0
        for i in range(max(0, x - radius), min(self.map_size, x + radius + 1)):
            for j in range(max(0, y - radius), min(self.map_size, y + radius + 1)):
                if (i - x) ** 2 + (j - y) ** 2 <= radius**2:
                    value += current_map[i, j]
        return value


def enhanced_randomized_worker_process(config, result_queue, epsilon_value, worker_id):
    """Enhanced worker process with better strategies."""
    np.random.seed(worker_id * 1000 + int(time.time()) % 1000)
    
    env = EnhancedRandomizedRadiiEnvironment(map_size=config.map_size)
    agent = EnhancedHeuristicAgent(map_size=config.map_size)
    
    episodes_completed = 0
    maps_generated = 0
    
    while True:
        try:
            # Get current epsilon
            with epsilon_value.get_lock():
                current_epsilon = epsilon_value.value
            
            # Run episode
            state = env.reset()
            maps_generated += 1
            
            episode_reward = 0
            episode_experiences = []
            total_coverage_improvement = 0
            
            # Store radii configuration for this episode
            radii_config = env.radii.copy()
            n_circles = len(radii_config)
            
            while True:
                # Get valid actions mask
                current_radius = env.radii[env.current_radius_idx] if env.current_radius_idx < len(env.radii) else env.radii[-1]
                valid_mask = np.ones((config.map_size, config.map_size), dtype=bool)
                valid_mask[:current_radius, :] = False
                valid_mask[-current_radius:, :] = False
                valid_mask[:, :current_radius] = False
                valid_mask[:, -current_radius:] = False
                
                # Add current radius and placed circles to state for heuristic agent
                state_with_info = state.copy()
                state_with_info["current_radius"] = current_radius
                state_with_info["placed_circles"] = env.placed_circles.copy()
                
                # Get action from enhanced heuristic agent
                action = agent.act(state_with_info, valid_mask.flatten(), current_epsilon)
                
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
            
            # Send results with additional metrics
            result_data = {
                "experiences": episode_experiences,
                "episode_reward": episode_reward,
                "coverage": info["coverage"],
                "total_coverage_improvement": total_coverage_improvement,
                "avg_coverage_improvement": total_coverage_improvement / max(len(episode_experiences), 1),
                "worker_id": worker_id,
                "episodes_completed": episodes_completed,
                "maps_generated": maps_generated,
                "radii_config": radii_config,
                "n_circles": n_circles,
                "map_diversity_stats": {
                    "map_mean": env.original_map.mean(),
                    "map_std": env.original_map.std(),
                    "map_max": env.original_map.max()
                },
                "efficiency_score": np.mean([exp[4]["efficiency"] for exp in episode_experiences if isinstance(exp[4], dict) and "efficiency" in exp[4]]),
                "touching_score": info.get("touching_bonus", 0),
            }
            
            result_queue.put(result_data)
            
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            continue


# Keep the same replay buffer from the original
class FixedThreadSafeReplayBuffer:
    """Thread-safe replay buffer - FIXED version."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.lock = threading.Lock()
    
    def push(self, experience):
        with self.lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            # FIXED: Use random indices instead of np.random.choice to avoid shape issues
            available_buffer = [exp for exp in self.buffer[:len(self.buffer)] if exp is not None]
            if len(available_buffer) < batch_size:
                return None
            indices = np.random.choice(len(available_buffer), batch_size, replace=False)
            return [available_buffer[i] for i in indices]
    
    def __len__(self):
        with self.lock:
            return len([exp for exp in self.buffer if exp is not None])


class EnhancedRandomizedRadiiTrainer:
    """Enhanced trainer with improvements."""
    
    def __init__(self, config: EnhancedRandomizedRadiiConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.next_breakpoint = 0
        
        # Initialize agent with adjusted parameters
        self.agent = GuidedDQNAgent(
            map_size=config.map_size,
            learning_rate=config.learning_rate,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay_steps=config.epsilon_decay_episodes,
            batch_size=config.batch_size,
            tau=0.005,  # Slightly higher for faster target updates
        )
        
        # Use mixed precision if available
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        
        # Shared epsilon value
        self.epsilon_value = mp.Value('f', config.epsilon_start)
        
        # Result queue for async communication
        self.result_queue = mp.Queue(maxsize=config.n_workers * 2)
        
        # Enhanced replay buffer
        self.replay_buffer = FixedThreadSafeReplayBuffer(config.buffer_size)
        
        # Start worker processes
        self.workers = []
        for i in range(config.n_workers):
            worker = mp.Process(
                target=enhanced_randomized_worker_process,
                args=(config, self.result_queue, self.epsilon_value, i),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_coverage = []
        self.coverage_improvements = []
        self.avg_coverage_improvements = []
        self.losses = []
        self.training_step = 0
        
        # Enhanced metrics
        self.efficiency_scores = []
        self.touching_scores = []
        
        # Randomized radii metrics
        self.radii_configs = []
        self.n_circles_history = []
        
        # Map diversity tracking
        self.map_diversity_stats = []
        self.worker_stats = {}
        
        # Calculate save frequencies based on total episodes
        n_model_saves = 20
        n_image_saves = 100
        
        # Periodic checkers with calculated frequencies
        model_save_freq = max(100, config.n_episodes // n_model_saves)
        image_save_freq = max(100, config.n_episodes // n_image_saves)
        eval_freq = max(100, config.n_episodes // 50)  # ~50 evaluations
        
        self.model_save_checker = RobustPeriodicChecker(model_save_freq)
        self.image_save_checker = RobustPeriodicChecker(image_save_freq)
        self.eval_checker = RobustPeriodicChecker(eval_freq)
        
        print(f"Save frequencies - Models: every {model_save_freq} episodes, Images: every {image_save_freq} episodes")
        
        # Create directories for organized saving
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        
        # Initialize periodic task trackers
        self.periodic_tasks = PeriodicTaskTracker()
        
        # Register periodic tasks
        self.periodic_tasks.register_task(
            'target_update', 
            self.config.target_update_freq,
            lambda step: self.agent.update_target_network()
        )
        
        # Use individual checkers for more control
        self.target_update_checker = RobustPeriodicChecker(self.config.target_update_freq)
        self.progress_checker = RobustPeriodicChecker(200)
        
        # Experience collection thread
        self.collection_thread = threading.Thread(target=self._collect_experiences, daemon=True)
        self.collection_thread.start()


@dataclass
class EnhancedRandomizedRadiiConfig:
    """Configuration for enhanced randomized radii training."""
    n_episodes: int = 100000
    n_workers: int = 32
    map_size: int = 128
    batch_size: int = 256  # Increased from 128
    buffer_size: int = 1000000  # Increased from 500000
    gradient_accumulation_steps: int = 4  # Increased from 2
    learning_rate: float = 5e-5  # Reduced from 1e-4
    epsilon_start: float = 0.8  # Reduced from 1.0
    epsilon_end: float = 0.05  # Increased from 0.01
    epsilon_decay_episodes: int = 50000  # Increased from 40000
    target_update_freq: int = 2000  # Increased from 1000
    visualize_every: int = 2000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"