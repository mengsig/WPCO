#!/usr/bin/env python3
"""
WORKING Randomized Radii Parallel Training
==========================================
Based on the proven coverage_aligned_parallel_train.py with randomized radii added.
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
)


class RandomizedRadiiEnvironment(AdvancedCirclePlacementEnv):
    """Environment with randomized radii configuration."""
    
    def __init__(self, map_size=64):
        self.previous_coverage = 0.0
        # Generate initial random radii - will be regenerated on each reset
        self._generate_random_radii()
        super().__init__(map_size, radii=self.radii)
        
    def _generate_random_radii(self):
        """Generate random radii configuration."""
        # Random number of circles (3-15)
        n_circles = np.random.randint(3, 16)
        
        # Generate random radii (2-20)
        radii = []
        for _ in range(n_circles):
            radius = np.random.randint(2, 21)
            radii.append(radius)
        
        # Sort descending for strategic placement
        self.radii = sorted(radii, reverse=True)
        
    def reset(self, weighted_matrix=None):
        """Reset environment with new random radii configuration."""
        # Generate new random radii for each episode
        self._generate_random_radii()
        
        if weighted_matrix is None:
            # Generate new random map
            weighted_matrix = random_seeder(self.map_size, time_steps=100000)
        
        # Reset environment state manually (avoid calling __init__ again)
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
            from src.algorithms.dqn_agent import HeatmapFeatureExtractor
            self.feature_extractor = HeatmapFeatureExtractor(self.map_size)
        
        self.previous_coverage = 0.0
        return self._get_enhanced_state()
    
    def step(self, action):
        """Step with simple, working reward function."""
        x, y = action
        radius = self.radii[self.current_radius_idx]
        
        # Store current coverage before action
        current_coverage = 1 - (self.current_map.sum() / (self.original_map.sum() + 1e-8))
        
        # Calculate base collection value
        included_weight = compute_included(self.current_map, x, y, radius)
        
        # Check for overlaps with existing circles
        overlap_penalty = 0.0
        for px, py, pr in self.placed_circles:
            distance = np.sqrt((x - px)**2 + (y - py)**2)
            min_distance = radius + pr
            if distance < min_distance:
                # Overlap detected
                overlap_ratio = (min_distance - distance) / min_distance
                overlap_penalty += overlap_ratio * 5.0  # Strong penalty
        
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
        
        # SIMPLE, WORKING REWARD FUNCTION
        reward = 0.0
        
        # 1. Basic reward: value collected
        reward += included_weight
        
        # 2. Coverage improvement bonus
        coverage_improvement = new_coverage - current_coverage
        if coverage_improvement > 0:
            reward += coverage_improvement * 100.0
        
        # 3. Overlap penalty
        reward -= overlap_penalty
        
        # 4. Boundary penalty
        edge_distance = min(x, y, self.map_size - x, self.map_size - y)
        if edge_distance < radius:
            boundary_penalty = (radius - edge_distance) / radius
            reward -= boundary_penalty * 2.0
        
        # Move to next radius
        self.current_radius_idx += 1
        done = self.current_radius_idx >= len(self.radii)
        
        # Final bonus for good coverage
        if done:
            if new_coverage > 0.8:
                reward += 50.0
            elif new_coverage > 0.6:
                reward += 20.0
            elif new_coverage > 0.4:
                reward += 10.0
            elif new_coverage < 0.2:
                reward -= 10.0
        
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
        }
        
        return self._get_enhanced_state(), reward, done, info


class SimpleHeuristicAgent:
    """Simple heuristic agent for worker processes."""
    
    def __init__(self, map_size=64):
        self.map_size = map_size
    
    def act(self, state_dict, valid_mask=None, epsilon=0.1):
        """Simple action selection."""
        current_radius = state_dict.get("current_radius", 5)
        current_map = state_dict["current_map"]
        
        if np.random.random() < epsilon:
            # Random exploration
            if valid_mask is not None:
                valid_positions = np.argwhere(valid_mask.reshape(self.map_size, self.map_size))
                if len(valid_positions) > 0:
                    idx = np.random.randint(len(valid_positions))
                    return tuple(valid_positions[idx])
            
            return (
                np.random.randint(current_radius, self.map_size - current_radius),
                np.random.randint(current_radius, self.map_size - current_radius)
            )
        else:
            # Greedy: find position with highest local value
            best_value = -1
            best_action = None
            
            # Sample some positions around high-value areas
            for _ in range(10):  # Limited sampling for speed
                # Find a high-value position
                high_value_positions = np.argwhere(current_map > np.percentile(current_map, 80))
                if len(high_value_positions) > 0:
                    center = high_value_positions[np.random.randint(len(high_value_positions))]
                    
                    # Sample around this center
                    for _ in range(3):
                        offset_x = np.random.randint(-current_radius, current_radius + 1)
                        offset_y = np.random.randint(-current_radius, current_radius + 1)
                        
                        x = np.clip(center[0] + offset_x, current_radius, self.map_size - current_radius - 1)
                        y = np.clip(center[1] + offset_y, current_radius, self.map_size - current_radius - 1)
                        
                        # Calculate local value
                        local_value = 0
                        for i in range(max(0, x - current_radius), min(self.map_size, x + current_radius + 1)):
                            for j in range(max(0, y - current_radius), min(self.map_size, y + current_radius + 1)):
                                if (i - x) ** 2 + (j - y) ** 2 <= current_radius**2:
                                    local_value += current_map[i, j]
                        
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


def randomized_radii_worker_process(config, result_queue, epsilon_value, worker_id):
    """Worker process for randomized radii training."""
    np.random.seed(worker_id * 1000 + int(time.time()) % 1000)
    
    env = RandomizedRadiiEnvironment(map_size=config.map_size)
    agent = SimpleHeuristicAgent(map_size=config.map_size)
    
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
                
                # Add current radius to state for heuristic agent
                state_with_radius = state.copy()
                state_with_radius["current_radius"] = current_radius
                
                # Get action from heuristic agent
                action = agent.act(state_with_radius, valid_mask.flatten(), current_epsilon)
                
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
                "maps_generated": maps_generated,
                "radii_config": radii_config,
                "n_circles": n_circles,
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


@dataclass
class RandomizedRadiiConfig:
    """Configuration for randomized radii training."""
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


class ThreadSafeReplayBuffer:
    """Thread-safe replay buffer."""
    
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
            return np.random.choice(self.buffer[:len(self.buffer)], batch_size, replace=False).tolist()
    
    def __len__(self):
        with self.lock:
            return len(self.buffer)


class WorkingRandomizedRadiiTrainer:
    """Working randomized radii trainer based on proven coverage_aligned architecture."""
    
    def __init__(self, config: RandomizedRadiiConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize agent
        self.agent = GuidedDQNAgent(
            map_size=config.map_size,
            learning_rate=config.learning_rate
        )
        
        # Use mixed precision if available
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
                target=randomized_radii_worker_process,
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
        
        # Randomized radii metrics
        self.radii_configs = []
        self.n_circles_history = []
        
        # Map diversity tracking
        self.map_diversity_stats = []
        self.worker_stats = {}
        
        # Experience collection thread
        self.collection_thread = threading.Thread(target=self._collect_experiences, daemon=True)
        self.collection_thread.start()
    
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
                self.coverage_improvements.append(result_data["total_coverage_improvement"])
                self.avg_coverage_improvements.append(result_data["avg_coverage_improvement"])
                
                # Record randomized radii metrics
                self.radii_configs.append(result_data["radii_config"])
                self.n_circles_history.append(result_data["n_circles"])
                
                # Track map diversity
                if "map_diversity_stats" in result_data:
                    self.map_diversity_stats.append(result_data["map_diversity_stats"])
                
                # Update worker statistics
                worker_id = result_data["worker_id"]
                self.worker_stats[worker_id] = {
                    "episodes_completed": result_data.get("episodes_completed", 0),
                    "maps_generated": result_data.get("maps_generated", 0)
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
        """Training step with proper error handling."""
        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return None
        
        try:
            state_batch, actions, rewards, next_states, dones = self._prepare_batch_tensors_optimized(batch)
            
            # Convert actions to indices
            action_indices = []
            for action in actions:
                if isinstance(action, (list, tuple)) and len(action) == 2:
                    x, y = action
                    idx = x * self.config.map_size + y
                else:
                    idx = action
                action_indices.append(idx)
            
            action_indices = torch.LongTensor(action_indices).to(self.device)
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    current_q_values = self.agent.q_network(state_batch)
                    
                    # Flatten if needed
                    if current_q_values.dim() == 3:
                        current_q_values = current_q_values.view(current_q_values.size(0), -1)
                    
                    current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
                    target_q_values = torch.FloatTensor(rewards).to(self.device)
                    loss = nn.MSELoss()(current_q_values, target_q_values)
                
                self.agent.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.agent.optimizer)
                self.scaler.update()
            else:
                current_q_values = self.agent.q_network(state_batch)
                
                # Flatten if needed
                if current_q_values.dim() == 3:
                    current_q_values = current_q_values.view(current_q_values.size(0), -1)
                
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
    
    def train(self):
        """Main training loop."""
        print("=" * 100)
        print(f"WORKING RANDOMIZED RADII TRAINING WITH {self.config.n_workers} WORKERS")
        print("=" * 100)
        print(f"Target episodes: {self.config.n_episodes:,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Buffer size: {self.config.buffer_size:,}")
        print(f"Workers: {self.config.n_workers}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Randomized radii: 3-15 circles, radii 2-20")
        print("=" * 100)
        
        pbar = tqdm(total=self.config.n_episodes, desc="Working Randomized Training")
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
                recent_reward = np.mean(self.episode_rewards[-1000:]) if len(self.episode_rewards) >= 1000 else np.mean(self.episode_rewards) if self.episode_rewards else 0
                recent_circles = np.mean(self.n_circles_history[-1000:]) if len(self.n_circles_history) >= 1000 else np.mean(self.n_circles_history) if self.n_circles_history else 0
                
                with self.epsilon_value.get_lock():
                    current_epsilon = self.epsilon_value.value
                
                pbar.set_postfix({
                    "Coverage": f"{recent_coverage:.1%}",
                    "Reward": f"{recent_reward:.2f}",
                    "AvgCircles": f"{recent_circles:.1f}",
                    "Buffer": f"{len(self.replay_buffer):,}",
                    "Loss": f"{np.mean(self.losses[-100:]):.4f}" if self.losses else "N/A",
                    "Epsilon": f"{current_epsilon:.3f}"
                })
            
            # Progress updates
            if current_episodes > 0 and current_episodes % 200 == 0:
                self._print_progress_update(current_episodes)
            
            # Periodic cleanup and saving
            if current_episodes % 500 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if current_episodes > 0 and current_episodes % 5000 == 0:
                self._save_checkpoint(current_episodes)
            
            time.sleep(0.01)
        
        pbar.close()
        
        # Final statistics
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n" + "=" * 100)
        print("WORKING RANDOMIZED RADII TRAINING COMPLETE!")
        print("=" * 100)
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        print(f"Episodes per second: {self.config.n_episodes / training_time:.2f}")
        print(f"Final average coverage: {np.mean(self.episode_coverage[-1000:]):.1%}")
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
        recent_rewards = np.mean(self.episode_rewards[-200:]) if len(self.episode_rewards) >= 200 else np.mean(self.episode_rewards)
        best_coverage = max(self.episode_coverage)
        
        # Configuration diversity stats
        recent_circles = self.n_circles_history[-200:] if len(self.n_circles_history) >= 200 else self.n_circles_history
        avg_circles = np.mean(recent_circles)
        circles_range = f"{min(recent_circles)}-{max(recent_circles)}"
        
        # Correlation
        correlation = 0.0
        if len(self.episode_rewards) > 100:
            correlation = np.corrcoef(self.episode_rewards[-500:], self.episode_coverage[-500:])[0, 1]
        
        with self.epsilon_value.get_lock():
            current_epsilon = self.epsilon_value.value
        
        print(f"\n🎯 Episode {episode:,} Progress (Working Randomized Radii):")
        print(f"   Coverage: {recent_coverage:.1%} (Best: {best_coverage:.1%})")
        print(f"   Reward: {recent_rewards:.2f}")
        print(f"   Reward-Coverage Correlation: {correlation:.3f}")
        print(f"   Epsilon: {current_epsilon:.3f}")
        print(f"   Training Steps: {self.training_step:,}")
        print(f"   Buffer: {len(self.replay_buffer):,}/{self.config.buffer_size:,}")
        print(f"   Configuration Diversity:")
        print(f"     • Avg Circles: {avg_circles:.1f} (Range: {circles_range})")
        
        # Worker stats
        if self.worker_stats:
            total_episodes = sum(stats.get("episodes_completed", 0) for stats in self.worker_stats.values())
            total_maps = sum(stats.get("maps_generated", 0) for stats in self.worker_stats.values())
            print(f"   Workers: {len(self.worker_stats)} active, {total_episodes:,} episodes, {total_maps:,} maps generated")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_path = f"working_randomized_checkpoint_episode_{episode}.pth"
        
        save_data = {
            "episode": episode,
            "model_state_dict": self.agent.q_network.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "training_step": self.training_step,
            "episode_rewards": self.episode_rewards[-1000:],
            "episode_coverage": self.episode_coverage[-1000:],
            "n_circles_history": self.n_circles_history[-1000:],
            "config": self.config
        }
        
        torch.save(save_data, checkpoint_path)
        print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self):
        """Save the trained model and metrics."""
        model_path = "working_randomized_radii_dqn_model.pth"
        
        save_data = {
            "model_state_dict": self.agent.q_network.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "training_step": self.training_step,
            "episode_rewards": self.episode_rewards,
            "episode_coverage": self.episode_coverage,
            "n_circles_history": self.n_circles_history,
            "config": self.config
        }
        
        torch.save(save_data, model_path)
        print(f"   💾 Final model saved: {model_path}")
    
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
    print(f"Using {n_workers} workers for working randomized radii training")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Configuration
    config = RandomizedRadiiConfig(
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
    trainer = WorkingRandomizedRadiiTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()