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
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.algorithms.dqn_agent import (
    AdvancedCirclePlacementEnv,
    GuidedDQNAgent,
    random_seeder,
)


@dataclass
class BulletproofConfig:
    """Bulletproof training configuration."""
    n_episodes: int = 100000
    n_workers: int = 32  # Conservative number
    batch_size: int = 64  # Safe batch size
    buffer_size: int = 200000  # Conservative buffer
    learning_rate: float = 1e-4
    map_size: int = 128
    visualize_every: int = 1000
    save_every: int = 5000
    target_update_freq: int = 100
    gradient_accumulation_steps: int = 4


class SimpleHeuristicAgent:
    """Simple heuristic agent that doesn't use neural networks."""
    
    def __init__(self, map_size: int):
        self.map_size = map_size
        
    def act(self, state_dict, env, valid_mask=None, epsilon=0.5):
        """Choose action using heuristics and exploration."""
        if np.random.random() < epsilon:
            # Exploration: random valid action
            if valid_mask is not None:
                valid_positions = np.argwhere(valid_mask > 0.5)
                if len(valid_positions) > 0:
                    # Weight by mask values for smarter exploration
                    weights = valid_mask[valid_positions[:, 0], valid_positions[:, 1]]
                    weights = weights / weights.sum()
                    idx = np.random.choice(len(valid_positions), p=weights)
                    return tuple(valid_positions[idx])
            
            return (np.random.randint(0, self.map_size), 
                   np.random.randint(0, self.map_size))
        else:
            # Exploitation: use environment heuristics
            suggestions = env.get_suggested_positions(n_suggestions=5)
            if suggestions:
                return suggestions[0]  # Take best suggestion
            
            # Fallback to high-value position
            current_map = state_dict["current_map"]
            max_pos = np.unravel_index(np.argmax(current_map), current_map.shape)
            return max_pos


def safe_worker_process(worker_id: int, config: BulletproofConfig,
                       control_queue: mp.Queue, result_queue: mp.Queue,
                       epsilon_value: mp.Value):
    """Completely safe worker process with no tensor serialization."""
    
    # Create environment and agent
    env = AdvancedCirclePlacementEnv(map_size=config.map_size)
    agent = SimpleHeuristicAgent(config.map_size)
    
    episodes_completed = 0
    
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
            
            # Run episode
            experiences = []
            episode_reward = 0
            
            while True:
                # Get valid actions
                valid_mask = env.get_valid_actions_mask()
                
                # Choose action
                action = agent.act(state, env, valid_mask, current_epsilon)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Convert states to serializable format (no tensors!)
                serializable_state = {
                    "current_map": state["current_map"].tolist() if hasattr(state["current_map"], 'tolist') else state["current_map"],
                    "placed_mask": state["placed_mask"].tolist() if hasattr(state["placed_mask"], 'tolist') else state["placed_mask"],
                    "value_density": state["value_density"].tolist() if hasattr(state["value_density"], 'tolist') else state["value_density"],
                    "features": state["features"].tolist() if hasattr(state["features"], 'tolist') else state["features"],
                }
                
                serializable_next_state = None
                if not done:
                    serializable_next_state = {
                        "current_map": next_state["current_map"].tolist() if hasattr(next_state["current_map"], 'tolist') else next_state["current_map"],
                        "placed_mask": next_state["placed_mask"].tolist() if hasattr(next_state["placed_mask"], 'tolist') else next_state["placed_mask"],
                        "value_density": next_state["value_density"].tolist() if hasattr(next_state["value_density"], 'tolist') else next_state["value_density"],
                        "features": next_state["features"].tolist() if hasattr(next_state["features"], 'tolist') else next_state["features"],
                    }
                
                # Store experience with serializable data
                experience = (serializable_state, action, reward, serializable_next_state, done)
                experiences.append(experience)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Send results (all basic Python types)
            result_data = {
                "worker_id": worker_id,
                "experiences": experiences,
                "episode_reward": episode_reward,
                "coverage": info["coverage"]
            }
            
            result_queue.put(result_data)
            episodes_completed += 1
            
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            time.sleep(1)  # Brief pause before retrying


class BulletproofReplayBuffer:
    """Simple replay buffer with no tensor storage."""
    
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


class BulletproofTrainer:
    """Bulletproof parallel trainer with no serialization issues."""
    
    def __init__(self, config: BulletproofConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Create main agent (only one GPU instance)
        self.agent = GuidedDQNAgent(
            map_size=config.map_size,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
        )
        
        # Mixed precision with updated API
        if torch.cuda.is_available():
            try:
                self.scaler = torch.amp.GradScaler('cuda')
            except:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Replay buffer
        self.replay_buffer = BulletproofReplayBuffer(config.buffer_size)
        
        # Shared epsilon value
        self.epsilon_value = mp.Value('d', 1.0)  # Shared double
        
        # Multiprocessing setup - no tensor sharing!
        self.control_queues = [mp.Queue() for _ in range(config.n_workers)]
        self.result_queue = mp.Queue(maxsize=config.n_workers * 2)
        
        # Start workers
        self.workers = []
        for i in range(config.n_workers):
            worker = mp.Process(
                target=safe_worker_process,
                args=(i, config, self.control_queues[i], self.result_queue, self.epsilon_value)
            )
            worker.start()
            self.workers.append(worker)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_coverage = []
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
                
            except:
                continue
    
    def _update_epsilon(self):
        """Update shared epsilon value."""
        # Epsilon decay
        epsilon_decay = 0.999995
        epsilon_min = 0.01
        
        with self.epsilon_value.get_lock():
            current_epsilon = self.epsilon_value.value
            new_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)
            self.epsilon_value.value = new_epsilon
            self.agent.epsilon = new_epsilon  # Update main agent too
    
    def _prepare_batch_tensors(self, batch):
        """Convert batch data to tensors safely."""
        states = [e[0] for e in batch]
        actions = [e[1] for e in batch]
        rewards = [e[2] for e in batch]
        next_states = [e[3] for e in batch if e[3] is not None]
        dones = [e[4] for e in batch]
        
        # Convert to numpy arrays first, then tensors
        current_maps = torch.FloatTensor([np.array(s["current_map"]) for s in states]).to(self.device)
        placed_masks = torch.FloatTensor([np.array(s["placed_mask"]) for s in states]).to(self.device)
        value_densities = torch.FloatTensor([np.array(s["value_density"]) for s in states]).to(self.device)
        features = torch.FloatTensor([np.array(s["features"]) for s in states]).to(self.device)
        
        state_batch = {
            "current_map": current_maps,
            "placed_mask": placed_masks,
            "value_density": value_densities,
            "features": features,
        }
        
        return state_batch, actions, rewards, next_states, dones
    
    def _train_step(self):
        """Perform one training step."""
        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return None
        
        try:
            # Prepare batch
            state_batch, actions, rewards, next_states, dones = self._prepare_batch_tensors(batch)
            
            # Forward pass
            current_q_values = self.agent.q_network(state_batch)
            
            # Convert actions to indices
            action_indices = torch.LongTensor([[a[0] * self.config.map_size + a[1]] for a in actions]).to(self.device)
            current_q_values = current_q_values.view(self.config.batch_size, -1).gather(1, action_indices).squeeze()
            
            # Target Q values
            next_q_values = torch.zeros(self.config.batch_size).to(self.device)
            if next_states:
                next_current_maps = torch.FloatTensor([np.array(s["current_map"]) for s in next_states]).to(self.device)
                next_placed_masks = torch.FloatTensor([np.array(s["placed_mask"]) for s in next_states]).to(self.device)
                next_value_densities = torch.FloatTensor([np.array(s["value_density"]) for s in next_states]).to(self.device)
                next_features = torch.FloatTensor([np.array(s["features"]) for s in next_states]).to(self.device)
                
                next_state_batch = {
                    "current_map": next_current_maps,
                    "placed_mask": next_placed_masks,
                    "value_density": next_value_densities,
                    "features": next_features,
                }
                
                non_final_mask = torch.tensor([not d for d in dones], dtype=torch.bool).to(self.device)
                
                with torch.no_grad():
                    # Double DQN
                    next_actions = self.agent.q_network(next_state_batch).view(len(next_states), -1).max(1)[1]
                    next_q_values[non_final_mask] = (
                        self.agent.target_network(next_state_batch)
                        .view(len(next_states), -1)
                        .gather(1, next_actions.unsqueeze(1))
                        .squeeze()
                    )
            
            # Compute loss
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            targets = rewards_tensor + self.agent.gamma * next_q_values
            loss = nn.functional.smooth_l1_loss(current_q_values, targets)
            
            # Backward pass with mixed precision
            self.agent.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss_scaled = loss
                self.scaler.scale(loss_scaled).backward()
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
            
            # Periodic GPU cleanup
            if self.training_step % 50 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return loss.item()
            
        except Exception as e:
            print(f"Training step error: {e}")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return None
    
    def visualize_strategy(self, episode, save_path="bulletproof_strategy.png"):
        """Visualize the agent's placement strategy."""
        # Create test environment
        test_env = AdvancedCirclePlacementEnv(self.config.map_size)
        weighted_matrix = random_seeder(self.config.map_size, time_steps=100000)
        state = test_env.reset(weighted_matrix)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original heatmap
        axes[0, 0].imshow(test_env.original_map, cmap="hot")
        axes[0, 0].set_title("Original Heatmap")
        axes[0, 0].axis("off")
        
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
            
            # Visualize at specific steps
            if i + 1 in steps_to_show and step_idx < 3:
                positions = [(0, 1), (0, 2), (1, 1)]
                row, col = positions[step_idx]
                
                axes[row, col].imshow(test_env.original_map, cmap="hot", alpha=0.6)
                
                # Draw circles
                for x, y, r in test_env.placed_circles:
                    circle = plt.Circle((y, x), r, fill=False, color="blue", linewidth=2)
                    axes[row, col].add_patch(circle)
                
                axes[row, col].set_title(f"After {i + 1} circles (Coverage: {info['coverage']:.1%})")
                axes[row, col].axis("off")
                
                step_idx += 1
            
            if done:
                break
        
        # Final result
        axes[1, 0].imshow(test_env.original_map, cmap="hot", alpha=0.6)
        for x, y, r in test_env.placed_circles:
            circle = plt.Circle((y, x), r, fill=False, color="blue", linewidth=2)
            axes[1, 0].add_patch(circle)
        axes[1, 0].set_title(f"Final (Coverage: {info['coverage']:.1%})")
        axes[1, 0].axis("off")
        
        # Feature visualization
        axes[1, 1].imshow(test_env.current_map, cmap="hot")
        axes[1, 1].set_title("Remaining Values")
        axes[1, 1].axis("off")
        
        # Performance stats
        axes[1, 2].text(0.1, 0.8, f"Episode: {episode}", fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.7, f"Coverage: {info['coverage']:.1%}", fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.6, f"Training Steps: {self.training_step}", fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.5, f"Buffer Size: {len(self.replay_buffer)}", fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].set_title("Training Stats")
        axes[1, 2].axis("off")
        
        plt.suptitle(f"Bulletproof Agent Strategy - Episode {episode}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        # Clear GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def train(self):
        """Bulletproof training loop."""
        print("=" * 100)
        print(f"BULLETPROOF PARALLEL TRAINING WITH {self.config.n_workers} WORKERS")
        print("=" * 100)
        print(f"Target episodes: {self.config.n_episodes:,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Buffer size: {self.config.buffer_size:,}")
        print(f"GPU Memory Management: Ultra-Safe")
        print("=" * 100)
        
        pbar = tqdm(total=self.config.n_episodes, desc="Bulletproof Training")
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
                
                with self.epsilon_value.get_lock():
                    current_epsilon = self.epsilon_value.value
                
                pbar.set_postfix({
                    "Coverage": f"{recent_coverage:.1%}",
                    "Reward": f"{recent_reward:.2f}",
                    "Buffer": f"{len(self.replay_buffer):,}",
                    "Loss": f"{np.mean(self.losses[-100:]):.4f}" if self.losses else "N/A",
                    "Epsilon": f"{current_epsilon:.3f}"
                })
            
            # Periodic evaluation and visualization
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
        print("BULLETPROOF TRAINING COMPLETE!")
        print("=" * 100)
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        print(f"Episodes per second: {self.config.n_episodes / training_time:.2f}")
        print(f"Final average coverage: {np.mean(self.episode_coverage[-1000:]):.1%}")
        print(f"Best coverage achieved: {max(self.episode_coverage):.1%}")
        print(f"Total training steps: {self.training_step:,}")
        
        self._save_final_model()
        self._cleanup()
    
    def _evaluate_and_visualize(self, episode: int):
        """Detailed evaluation and visualization."""
        print(f"\n{'=' * 80}")
        print(f"Episode {episode:,}/{self.config.n_episodes:,}")
        print(f"{'=' * 80}")
        
        # Performance stats
        recent_coverage = np.mean(self.episode_coverage[-1000:])
        recent_rewards = np.mean(self.episode_rewards[-1000:])
        best_coverage = max(self.episode_coverage)
        
        print(f"\nPERFORMANCE:")
        print(f"  Recent avg coverage: {recent_coverage:.1%}")
        print(f"  Best coverage: {best_coverage:.1%}")
        print(f"  Recent avg reward: {recent_rewards:.2f}")
        print(f"  Training steps: {self.training_step:,}")
        print(f"  Buffer size: {len(self.replay_buffer):,}")
        
        if self.losses:
            print(f"\nTRAINING:")
            print(f"  Recent avg loss: {np.mean(self.losses[-1000:]):.4f}")
            with self.epsilon_value.get_lock():
                print(f"  Epsilon: {self.epsilon_value.value:.3f}")
        
        # Visualize strategy
        self.visualize_strategy(episode, f"bulletproof_strategy_ep{episode}.png")
        print(f"\nStrategy visualization saved!")
        
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
                    "std": np.std(self.episode_coverage[-1000:])
                }
            }, f"bulletproof_model_ep{episode}.pth")
            print(f"Checkpoint saved: bulletproof_model_ep{episode}.pth")
    
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
            "losses": self.losses,
            "config": self.config,
        }, "bulletproof_final_model.pth")
        
        # Plot results
        self._plot_training_progress()
    
    def _plot_training_progress(self):
        """Plot comprehensive training progress."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Coverage over time
        ax1.plot(self.episode_coverage, alpha=0.3, label="Episode")
        if len(self.episode_coverage) > 1000:
            smoothed = np.convolve(self.episode_coverage, np.ones(1000)/1000, mode='valid')
            ax1.plot(smoothed, linewidth=2, label='1000-episode average')
        ax1.set_title('Coverage Over Training')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Coverage')
        ax1.legend()
        ax1.grid(True)
        
        # Rewards over time
        ax2.plot(self.episode_rewards, alpha=0.3, label="Episode")
        if len(self.episode_rewards) > 1000:
            smoothed = np.convolve(self.episode_rewards, np.ones(1000)/1000, mode='valid')
            ax2.plot(smoothed, linewidth=2, label='1000-episode average')
        ax2.set_title('Rewards Over Training')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True)
        
        # Loss over time
        if self.losses:
            ax3.plot(self.losses, alpha=0.5)
            if len(self.losses) > 1000:
                smoothed = np.convolve(self.losses, np.ones(1000)/1000, mode='valid')
                ax3.plot(smoothed, linewidth=2, label='1000-step average')
            ax3.set_title('Training Loss')
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Loss')
            ax3.legend()
            ax3.grid(True)
        
        # Coverage histogram
        ax4.hist(self.episode_coverage, bins=100, alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(self.episode_coverage), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.episode_coverage):.1%}')
        ax4.set_title('Coverage Distribution')
        ax4.set_xlabel('Coverage')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True)
        
        plt.suptitle(f'Bulletproof Training Results - {len(self.episode_rewards):,} Episodes')
        plt.tight_layout()
        plt.savefig('bulletproof_training_results.png', dpi=150, bbox_inches='tight')
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
        
        # Clear GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("Cleanup complete.")


def main():
    """Main bulletproof training function."""
    # System detection
    try:
        n_cores = mp.cpu_count()
    except:
        n_cores = 16
    
    # Very conservative worker count
    n_workers = min(32, max(8, n_cores - 4))
    
    print(f"System: {n_cores} cores")
    print(f"Using {n_workers} workers for bulletproof training")
    
    config = BulletproofConfig(
        n_episodes=100000,
        n_workers=n_workers,
        batch_size=64,
        buffer_size=200000,
        learning_rate=1e-4,
        visualize_every=1000,
        save_every=5000,
    )
    
    # Set multiprocessing method
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass
    
    trainer = BulletproofTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()