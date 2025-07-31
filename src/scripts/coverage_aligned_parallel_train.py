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


class CoverageAlignedEnvironment(AdvancedCirclePlacementEnv):
    """Environment with coverage-aligned reward function."""
    
    def __init__(self, map_size=128):
        super().__init__(map_size)
        self.previous_coverage = 0.0
        
    def reset(self, weighted_matrix):
        """Reset environment and coverage tracking."""
        result = super().reset(weighted_matrix)
        self.previous_coverage = 0.0
        return result
    
    def step(self, action):
        """Step with coverage-aligned reward function."""
        x, y = action
        radius = self.radii[self.current_radius_idx]
        
        # Store current coverage before action
        current_coverage = 1 - (self.current_map.sum() / self.original_map.sum())
        
        # Calculate base collection value
        included_weight = compute_included(self.current_map, x, y, radius)
        
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
        new_coverage = 1 - (self.current_map.sum() / self.original_map.sum())
        
        # COVERAGE-ALIGNED REWARD FUNCTION
        # Primary reward: direct coverage improvement
        coverage_improvement = new_coverage - current_coverage
        reward = coverage_improvement * 10.0  # Scale up for learning
        
        # Bonus rewards that support coverage
        if included_weight > 0:
            # 1. Efficiency bonus - reward for good space utilization
            circle_area = np.pi * radius * radius
            efficiency = included_weight / (circle_area * self.original_map.max())
            reward += efficiency * 0.5
            
            # 2. Density bonus - reward for targeting high-value areas
            if self.original_map.max() > 0:
                density_ratio = included_weight / (circle_area * self.original_map.max())
                reward += density_ratio * 0.3
        else:
            # Penalty for placing circles that collect nothing
            reward -= 0.1
        
        # Move to next radius
        self.current_radius_idx += 1
        done = self.current_radius_idx >= len(self.radii)
        
        # Update coverage tracking
        self.coverage_history.append(new_coverage)
        self.previous_coverage = new_coverage
        
        info = {
            "coverage": new_coverage,
            "coverage_improvement": coverage_improvement,
            "included_weight": included_weight,
            "efficiency": efficiency if included_weight > 0 else 0,
        }
        
        if done:
            # Final bonus based on total coverage achieved
            if new_coverage > 0.9:
                reward += 5.0  # Excellent coverage
            elif new_coverage > 0.8:
                reward += 3.0  # Good coverage
            elif new_coverage > 0.7:
                reward += 1.0  # Decent coverage
            elif new_coverage > 0.6:
                reward += 0.5  # Acceptable coverage
            # No bonus for coverage < 60%
            
            return None, reward, done, info
        
        return self._get_enhanced_state(), reward, done, info


@dataclass
class CoverageAlignedConfig:
    """Configuration for coverage-aligned training."""
    n_episodes: int = 100000
    n_workers: int = 32
    batch_size: int = 64
    buffer_size: int = 200000
    learning_rate: float = 1e-4
    map_size: int = 128
    visualize_every: int = 1000
    save_every: int = 5000
    target_update_freq: int = 100
    gradient_accumulation_steps: int = 4
    # Fixed epsilon decay parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 20000


class CoverageHeuristicAgent:
    """Heuristic agent focused on coverage maximization."""
    
    def __init__(self, map_size: int):
        self.map_size = map_size
        
    def act(self, state_dict, env, valid_mask=None, epsilon=0.5):
        """Choose action prioritizing coverage improvement."""
        if np.random.random() < epsilon:
            # Smart exploration - prefer high-value areas
            if valid_mask is not None:
                valid_positions = np.argwhere(valid_mask > 0.5)
                if len(valid_positions) > 0:
                    # Weight by potential coverage improvement
                    current_map = state_dict["current_map"]
                    radius = env.radii[env.current_radius_idx]
                    
                    weights = []
                    for pos in valid_positions:
                        x, y = pos
                        # Estimate coverage improvement for this position
                        local_value = 0
                        for i in range(max(0, x - radius), min(self.map_size, x + radius + 1)):
                            for j in range(max(0, y - radius), min(self.map_size, y + radius + 1)):
                                if (i - x) ** 2 + (j - y) ** 2 <= radius**2:
                                    local_value += current_map[i, j]
                        weights.append(local_value)
                    
                    if sum(weights) > 0:
                        weights = np.array(weights)
                        weights = weights / weights.sum()
                        idx = np.random.choice(len(valid_positions), p=weights)
                        return tuple(valid_positions[idx])
            
            return (np.random.randint(0, self.map_size), 
                   np.random.randint(0, self.map_size))
        else:
            # Greedy: choose position that maximizes coverage improvement
            current_map = state_dict["current_map"]
            radius = env.radii[env.current_radius_idx]
            
            best_value = -1
            best_action = None
            
            # Sample some positions to find the best
            for _ in range(20):  # Sample 20 positions
                x = np.random.randint(radius, self.map_size - radius)
                y = np.random.randint(radius, self.map_size - radius)
                
                # Calculate potential coverage improvement
                local_value = 0
                for i in range(max(0, x - radius), min(self.map_size, x + radius + 1)):
                    for j in range(max(0, y - radius), min(self.map_size, y + radius + 1)):
                        if (i - x) ** 2 + (j - y) ** 2 <= radius**2:
                            local_value += current_map[i, j]
                
                if local_value > best_value:
                    best_value = local_value
                    best_action = (x, y)
            
            if best_action is not None:
                return best_action
            
            # Fallback
            return (np.random.randint(0, self.map_size), 
                   np.random.randint(0, self.map_size))


def coverage_aligned_worker_process(worker_id: int, config: CoverageAlignedConfig,
                                   control_queue: mp.Queue, result_queue: mp.Queue,
                                   epsilon_value: mp.Value):
    """Worker process with coverage-aligned environment."""
    
    # Create coverage-aligned environment and agent
    env = CoverageAlignedEnvironment(map_size=config.map_size)
    agent = CoverageHeuristicAgent(config.map_size)
    
    while True:
        try:
            # Check for stop signal
            try:
                signal = control_queue.get_nowait()
                if signal == "STOP":
                    break
            except:
                pass
            
            # Get current epsilon
            with epsilon_value.get_lock():
                current_epsilon = epsilon_value.value
            
            # Generate new map
            weighted_matrix = random_seeder(config.map_size, time_steps=100000)
            state = env.reset(weighted_matrix)
            
            # Pre-allocate lists for efficiency
            experiences = []
            episode_reward = 0
            coverage_improvements = []
            
            while True:
                # Get valid actions
                valid_mask = env.get_valid_actions_mask()
                
                # Choose action
                action = agent.act(state, env, valid_mask, current_epsilon)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Track coverage improvements
                coverage_improvements.append(info.get("coverage_improvement", 0))
                
                # Efficiently convert states to numpy arrays
                current_state_arrays = {
                    "current_map": np.array(state["current_map"], dtype=np.float32),
                    "placed_mask": np.array(state["placed_mask"], dtype=np.float32),
                    "value_density": np.array(state["value_density"], dtype=np.float32),
                    "features": np.array(state["features"], dtype=np.float32),
                }
                
                next_state_arrays = None
                if not done:
                    next_state_arrays = {
                        "current_map": np.array(next_state["current_map"], dtype=np.float32),
                        "placed_mask": np.array(next_state["placed_mask"], dtype=np.float32),
                        "value_density": np.array(next_state["value_density"], dtype=np.float32),
                        "features": np.array(next_state["features"], dtype=np.float32),
                    }
                
                # Store experience with numpy arrays
                experience = (current_state_arrays, action, reward, next_state_arrays, done)
                experiences.append(experience)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Send results efficiently
            result_data = {
                "worker_id": worker_id,
                "experiences": experiences,
                "episode_reward": episode_reward,
                "coverage": info["coverage"],
                "total_coverage_improvement": sum(coverage_improvements),
                "avg_coverage_improvement": np.mean(coverage_improvements) if coverage_improvements else 0,
            }
            
            result_queue.put(result_data)
            
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            time.sleep(1)


class OptimizedReplayBuffer:
    """Optimized replay buffer with efficient storage."""
    
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer = []
        self.position = 0
        
    def push(self, experience):
        """Add experience to buffer."""
        if len(self.buffer) < self.maxlen:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.maxlen
    
    def sample(self, batch_size: int):
        """Sample batch from buffer."""
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class CoverageAlignedTrainer:
    """Trainer with coverage-aligned reward function."""
    
    def __init__(self, config: CoverageAlignedConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Create main agent
        self.agent = GuidedDQNAgent(
            map_size=config.map_size,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
        )
        
        # Mixed precision
        if torch.cuda.is_available():
            try:
                self.scaler = torch.amp.GradScaler('cuda')
                self.use_amp = True
            except:
                try:
                    self.scaler = torch.cuda.amp.GradScaler()
                    self.use_amp = True
                except:
                    self.scaler = None
                    self.use_amp = False
        else:
            self.scaler = None
            self.use_amp = False
        
        # Replay buffer
        self.replay_buffer = OptimizedReplayBuffer(config.buffer_size)
        
        # Shared epsilon
        self.epsilon_value = mp.Value('d', config.epsilon_start)
        
        # Multiprocessing setup
        self.control_queues = [mp.Queue() for _ in range(config.n_workers)]
        self.result_queue = mp.Queue(maxsize=config.n_workers * 2)
        
        # Start workers
        self.workers = []
        for i in range(config.n_workers):
            worker = mp.Process(
                target=coverage_aligned_worker_process,
                args=(i, config, self.control_queues[i], self.result_queue, self.epsilon_value)
            )
            worker.start()
            self.workers.append(worker)
        
        # Training metrics - now tracking coverage improvements
        self.episode_rewards = []
        self.episode_coverage = []
        self.coverage_improvements = []
        self.avg_coverage_improvements = []
        self.losses = []
        self.training_step = 0
        
        # Experience collection thread
        self.collection_thread = threading.Thread(target=self._collect_experiences, daemon=True)
        self.collection_thread.start()
    
    def _collect_experiences(self):
        """Collect experiences from workers."""
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
                
            except:
                continue
    
    def _update_epsilon(self):
        """Update shared epsilon value with proper decay rate."""
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
        
        # Convert to single numpy array first, then to tensor
        current_maps_np = np.stack([s["current_map"] for s in states])
        placed_masks_np = np.stack([s["placed_mask"] for s in states])
        value_densities_np = np.stack([s["value_density"] for s in states])
        features_np = np.stack([s["features"] for s in states])
        
        # Convert numpy arrays to tensors
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
        """Optimized training step."""
        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return None
        
        try:
            state_batch, actions, rewards, next_states, dones = self._prepare_batch_tensors_optimized(batch)
            
            # Forward pass
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    current_q_values = self.agent.q_network(state_batch)
            else:
                current_q_values = self.agent.q_network(state_batch)
            
            # Convert actions to indices
            action_indices = torch.tensor([[a[0] * self.config.map_size + a[1]] for a in actions], 
                                        dtype=torch.long, device=self.device)
            current_q_values = current_q_values.view(self.config.batch_size, -1).gather(1, action_indices).squeeze()
            
            # Target Q values
            next_q_values = torch.zeros(self.config.batch_size, device=self.device)
            if next_states:
                next_current_maps_np = np.stack([s["current_map"] for s in next_states])
                next_placed_masks_np = np.stack([s["placed_mask"] for s in next_states])
                next_value_densities_np = np.stack([s["value_density"] for s in next_states])
                next_features_np = np.stack([s["features"] for s in next_states])
                
                next_current_maps = torch.from_numpy(next_current_maps_np).float().to(self.device)
                next_placed_masks = torch.from_numpy(next_placed_masks_np).float().to(self.device)
                next_value_densities = torch.from_numpy(next_value_densities_np).float().to(self.device)
                next_features = torch.from_numpy(next_features_np).float().to(self.device)
                
                next_state_batch = {
                    "current_map": next_current_maps,
                    "placed_mask": next_placed_masks,
                    "value_density": next_value_densities,
                    "features": next_features,
                }
                
                non_final_mask = torch.tensor([not d for d in dones], dtype=torch.bool, device=self.device)
                
                with torch.no_grad():
                    next_actions = self.agent.q_network(next_state_batch).view(len(next_states), -1).max(1)[1]
                    next_q_values[non_final_mask] = (
                        self.agent.target_network(next_state_batch)
                        .view(len(next_states), -1)
                        .gather(1, next_actions.unsqueeze(1))
                        .squeeze()
                    )
            
            # Compute loss
            rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)
            targets = rewards_tensor + self.agent.gamma * next_q_values
            loss = nn.functional.smooth_l1_loss(current_q_values, targets)
            
            # Backward pass
            self.agent.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.agent.optimizer)
                torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 1.0)
                self.scaler.step(self.agent.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 1.0)
                self.agent.optimizer.step()
            
            # Update target network
            if self.training_step % self.config.target_update_freq == 0:
                for target_param, param in zip(self.agent.target_network.parameters(), self.agent.q_network.parameters()):
                    target_param.data.copy_(self.agent.tau * param.data + (1.0 - self.agent.tau) * target_param.data)
            
            self.training_step += 1
            
            # Periodic cleanup
            if self.training_step % 100 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return loss.item()
            
        except Exception as e:
            print(f"Training step error: {e}")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return None
    
    def visualize_strategy(self, episode, save_path="coverage_aligned_strategy.png"):
        """Visualize the coverage-aligned agent's strategy."""
        # Use coverage-aligned environment for testing
        test_env = CoverageAlignedEnvironment(self.config.map_size)
        weighted_matrix = random_seeder(self.config.map_size, time_steps=100000)
        state = test_env.reset(weighted_matrix)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original heatmap
        axes[0, 0].imshow(test_env.original_map, cmap="hot")
        axes[0, 0].set_title("Original Heatmap")
        axes[0, 0].axis("off")
        
        # Track coverage improvements
        coverage_history = [0.0]
        
        # Show placements step by step
        steps_to_show = [2, 5, 8]
        step_idx = 0
        
        for i in range(len(test_env.radii)):
            valid_mask = test_env.get_valid_actions_mask()
            
            # Get agent's action
            with torch.no_grad():
                state_batch = self.agent._prepare_state_batch([state])
                q_values = self.agent.q_network(state_batch).squeeze(0)
                
                if valid_mask is not None:
                    mask_tensor = torch.FloatTensor(valid_mask).to(self.device)
                    q_values = q_values + (mask_tensor - 1) * 1e10
                
                action_idx = q_values.view(-1).argmax().item()
                action = (action_idx // self.config.map_size, action_idx % self.config.map_size)
            
            # Take step
            state, reward, done, info = test_env.step(action)
            coverage_history.append(info["coverage"])
            
            # Visualize at specific steps
            if i + 1 in steps_to_show and step_idx < 3:
                positions = [(0, 1), (0, 2), (1, 1)]
                row, col = positions[step_idx]
                
                axes[row, col].imshow(test_env.original_map, cmap="hot", alpha=0.6)
                
                # Draw circles
                for x, y, r in test_env.placed_circles:
                    circle = plt.Circle((y, x), r, fill=False, color="blue", linewidth=2)
                    axes[row, col].add_patch(circle)
                
                coverage_improvement = coverage_history[-1] - coverage_history[-2] if len(coverage_history) > 1 else 0
                axes[row, col].set_title(f"Step {i + 1}: Coverage {info['coverage']:.1%} (+{coverage_improvement:.1%})")
                axes[row, col].axis("off")
                
                step_idx += 1
            
            if done:
                break
        
        # Final result
        axes[1, 0].imshow(test_env.original_map, cmap="hot", alpha=0.6)
        for x, y, r in test_env.placed_circles:
            circle = plt.Circle((y, x), r, fill=False, color="blue", linewidth=2)
            axes[1, 0].add_patch(circle)
        axes[1, 0].set_title(f"Final Coverage: {info['coverage']:.1%}")
        axes[1, 0].axis("off")
        
        # Coverage progress
        axes[1, 1].plot(coverage_history, 'g-', linewidth=2, marker='o')
        axes[1, 1].set_title("Coverage Progress")
        axes[1, 1].set_xlabel("Circle Placement")
        axes[1, 1].set_ylabel("Coverage")
        axes[1, 1].grid(True)
        
        # Performance stats
        axes[1, 2].text(0.1, 0.9, f"Episode: {episode}", fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.8, f"Final Coverage: {info['coverage']:.1%}", fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.7, f"Training Steps: {self.training_step}", fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.6, f"Buffer Size: {len(self.replay_buffer)}", fontsize=12, transform=axes[1, 2].transAxes)
        
        # Show reward-coverage correlation
        if len(self.episode_rewards) > 100:
            recent_rewards = self.episode_rewards[-100:]
            recent_coverage = self.episode_coverage[-100:]
            correlation = np.corrcoef(recent_rewards, recent_coverage)[0, 1]
            axes[1, 2].text(0.1, 0.5, f"Reward-Coverage Corr: {correlation:.3f}", fontsize=12, transform=axes[1, 2].transAxes)
        
        axes[1, 2].set_title("Training Stats")
        axes[1, 2].axis("off")
        
        plt.suptitle(f"Coverage-Aligned Agent Strategy - Episode {episode}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def train(self):
        """Coverage-aligned training loop."""
        print("=" * 100)
        print(f"COVERAGE-ALIGNED PARALLEL TRAINING WITH {self.config.n_workers} WORKERS")
        print("=" * 100)
        print(f"Target episodes: {self.config.n_episodes:,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Buffer size: {self.config.buffer_size:,}")
        print(f"Epsilon decay: {self.config.epsilon_start} → {self.config.epsilon_end} over {self.config.epsilon_decay_episodes:,} episodes")
        print(f"Reward Function: Coverage-Aligned (reward ∝ coverage improvement)")
        print("=" * 100)
        
        pbar = tqdm(total=self.config.n_episodes, desc="Coverage-Aligned Training")
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
                recent_cov_imp = np.mean(self.avg_coverage_improvements[-1000:]) if len(self.avg_coverage_improvements) >= 1000 else np.mean(self.avg_coverage_improvements) if self.avg_coverage_improvements else 0
                
                with self.epsilon_value.get_lock():
                    current_epsilon = self.epsilon_value.value
                
                pbar.set_postfix({
                    "Coverage": f"{recent_coverage:.1%}",
                    "Reward": f"{recent_reward:.2f}",
                    "CovImp": f"{recent_cov_imp:.3f}",
                    "Buffer": f"{len(self.replay_buffer):,}",
                    "Loss": f"{np.mean(self.losses[-100:]):.4f}" if self.losses else "N/A",
                    "Epsilon": f"{current_epsilon:.3f}"
                })
            
            # Periodic evaluation
            if current_episodes > 0 and current_episodes % self.config.visualize_every == 0:
                self._evaluate_and_visualize(current_episodes)
            
            # Periodic cleanup
            if current_episodes % 500 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            time.sleep(0.01)
        
        pbar.close()
        
        # Final statistics
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n" + "=" * 100)
        print("COVERAGE-ALIGNED TRAINING COMPLETE!")
        print("=" * 100)
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        print(f"Episodes per second: {self.config.n_episodes / training_time:.2f}")
        print(f"Final average coverage: {np.mean(self.episode_coverage[-1000:]):.1%}")
        print(f"Best coverage achieved: {max(self.episode_coverage):.1%}")
        print(f"Total training steps: {self.training_step:,}")
        
        # Show reward-coverage correlation
        if len(self.episode_rewards) > 100:
            correlation = np.corrcoef(self.episode_rewards[-1000:], self.episode_coverage[-1000:])[0, 1]
            print(f"Final reward-coverage correlation: {correlation:.3f}")
        
        self._save_final_model()
        self._cleanup()
    
    def _evaluate_and_visualize(self, episode: int):
        """Evaluation with coverage-alignment metrics."""
        print(f"\n{'=' * 80}")
        print(f"Episode {episode:,}/{self.config.n_episodes:,}")
        print(f"{'=' * 80}")
        
        # Performance stats
        recent_coverage = np.mean(self.episode_coverage[-1000:]) if len(self.episode_coverage) >= 1000 else np.mean(self.episode_coverage)
        recent_rewards = np.mean(self.episode_rewards[-1000:]) if len(self.episode_rewards) >= 1000 else np.mean(self.episode_rewards)
        recent_cov_imp = np.mean(self.avg_coverage_improvements[-1000:]) if len(self.avg_coverage_improvements) >= 1000 else np.mean(self.avg_coverage_improvements)
        best_coverage = max(self.episode_coverage)
        
        print(f"\nPERFORMANCE:")
        print(f"  Recent avg coverage: {recent_coverage:.1%}")
        print(f"  Best coverage: {best_coverage:.1%}")
        print(f"  Recent avg reward: {recent_rewards:.2f}")
        print(f"  Recent avg coverage improvement: {recent_cov_imp:.3f}")
        print(f"  Training steps: {self.training_step:,}")
        print(f"  Buffer size: {len(self.replay_buffer):,}")
        
        # Reward-coverage alignment
        if len(self.episode_rewards) > 100:
            correlation = np.corrcoef(self.episode_rewards[-1000:], self.episode_coverage[-1000:])[0, 1]
            print(f"\nALIGNMENT:")
            print(f"  Reward-Coverage correlation: {correlation:.3f}")
            if correlation > 0.7:
                print(f"  ✅ Strong positive correlation - rewards aligned with coverage!")
            elif correlation > 0.3:
                print(f"  ⚠️  Moderate correlation - some alignment")
            else:
                print(f"  ❌ Weak correlation - rewards not well aligned")
        
        if self.losses:
            print(f"\nTRAINING:")
            print(f"  Recent avg loss: {np.mean(self.losses[-1000:]):.4f}")
            with self.epsilon_value.get_lock():
                print(f"  Epsilon: {self.epsilon_value.value:.3f}")
        
        # Visualize strategy
        try:
            self.visualize_strategy(episode, f"coverage_aligned_strategy_ep{episode}.png")
            print(f"\nStrategy visualization saved!")
        except Exception as e:
            print(f"Visualization error: {e}")
        
        # Save checkpoint
        if episode % self.config.save_every == 0:
            torch.save({
                "model_state_dict": self.agent.q_network.state_dict(),
                "target_state_dict": self.agent.target_network.state_dict(),
                "episode": episode,
                "training_step": self.training_step,
                "coverage_stats": {
                    "recent": recent_coverage,
                    "best": best_coverage,
                    "correlation": correlation if len(self.episode_rewards) > 100 else 0,
                }
            }, f"coverage_aligned_model_ep{episode}.pth")
            print(f"Checkpoint saved: coverage_aligned_model_ep{episode}.pth")
    
    def _save_final_model(self):
        """Save final model and statistics."""
        torch.save({
            "model_state_dict": self.agent.q_network.state_dict(),
            "target_state_dict": self.agent.target_network.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "episode": len(self.episode_rewards),
            "training_step": self.training_step,
            "episode_rewards": self.episode_rewards,
            "episode_coverage": self.episode_coverage,
            "coverage_improvements": self.coverage_improvements,
            "avg_coverage_improvements": self.avg_coverage_improvements,
            "losses": self.losses,
            "config": self.config,
        }, "coverage_aligned_final_model.pth")
        
        self._plot_training_progress()
    
    def _plot_training_progress(self):
        """Plot comprehensive training progress with coverage alignment."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Coverage over time
        ax1.plot(self.episode_coverage, alpha=0.3, label="Episode Coverage", color='green')
        if len(self.episode_coverage) > 1000:
            smoothed = np.convolve(self.episode_coverage, np.ones(1000)/1000, mode='valid')
            ax1.plot(smoothed, linewidth=2, label='1000-episode average', color='darkgreen')
        ax1.set_title('Coverage Over Training')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Coverage')
        ax1.legend()
        ax1.grid(True)
        
        # Rewards over time
        ax2.plot(self.episode_rewards, alpha=0.3, label="Episode Reward", color='blue')
        if len(self.episode_rewards) > 1000:
            smoothed = np.convolve(self.episode_rewards, np.ones(1000)/1000, mode='valid')
            ax2.plot(smoothed, linewidth=2, label='1000-episode average', color='darkblue')
        ax2.set_title('Rewards Over Training')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True)
        
        # Reward vs Coverage scatter plot
        if len(self.episode_rewards) > 100:
            sample_size = min(5000, len(self.episode_rewards))
            indices = np.random.choice(len(self.episode_rewards), sample_size, replace=False)
            rewards_sample = [self.episode_rewards[i] for i in indices]
            coverage_sample = [self.episode_coverage[i] for i in indices]
            
            ax3.scatter(coverage_sample, rewards_sample, alpha=0.5, s=1)
            
            # Add correlation line
            correlation = np.corrcoef(rewards_sample, coverage_sample)[0, 1]
            z = np.polyfit(coverage_sample, rewards_sample, 1)
            p = np.poly1d(z)
            coverage_range = np.linspace(min(coverage_sample), max(coverage_sample), 100)
            ax3.plot(coverage_range, p(coverage_range), "r--", alpha=0.8, linewidth=2)
            
            ax3.set_title(f'Reward vs Coverage (r={correlation:.3f})')
            ax3.set_xlabel('Coverage')
            ax3.set_ylabel('Reward')
            ax3.grid(True)
        
        # Coverage improvements over time
        if self.avg_coverage_improvements:
            ax4.plot(self.avg_coverage_improvements, alpha=0.5, color='orange')
            if len(self.avg_coverage_improvements) > 1000:
                smoothed = np.convolve(self.avg_coverage_improvements, np.ones(1000)/1000, mode='valid')
                ax4.plot(smoothed, linewidth=2, label='1000-episode average', color='darkorange')
            ax4.set_title('Average Coverage Improvement per Episode')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Avg Coverage Improvement')
            ax4.legend()
            ax4.grid(True)
        
        plt.suptitle(f'Coverage-Aligned Training Results - {len(self.episode_rewards):,} Episodes')
        plt.tight_layout()
        plt.savefig('coverage_aligned_training_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _cleanup(self):
        """Cleanup resources."""
        print("Cleaning up...")
        
        # Stop workers
        for q in self.control_queues:
            try:
                q.put("STOP")
            except:
                pass
        
        # Wait for workers
        for worker in self.workers:
            worker.join(timeout=3)
            if worker.is_alive():
                worker.terminate()
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("Cleanup complete.")


def main():
    """Main coverage-aligned training function."""
    try:
        n_cores = mp.cpu_count()
    except:
        n_cores = 32
    
    # Optimized worker count
    n_workers = min(64, max(16, n_cores - 8))
    
    print(f"System: {n_cores} cores")
    print(f"Using {n_workers} workers for coverage-aligned training")
    
    config = CoverageAlignedConfig(
        n_episodes=100000,
        n_workers=n_workers,
        batch_size=64,
        buffer_size=200000,
        learning_rate=1e-4,
        visualize_every=1000,
        save_every=5000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_episodes=20000,
    )
    
    # Set multiprocessing method
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass
    
    trainer = CoverageAlignedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()