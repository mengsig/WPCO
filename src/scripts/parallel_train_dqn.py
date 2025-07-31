import sys
import os
import multiprocessing as mp
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.algorithms.dqn_agent import (
    AdvancedCirclePlacementEnv,
    GuidedDQNAgent,
    random_seeder,
)
from src.utils.plotting_utils import plot_heatmap_with_circles


@dataclass
class TrainingConfig:
    """Configuration for parallel training."""
    n_episodes: int = 50000
    n_workers: int = 64  # Number of parallel environment workers
    batch_size: int = 64  # Larger batch size for better GPU utilization
    buffer_size: int = 200000  # Larger buffer for more diverse experiences
    learning_rate: float = 1e-4
    map_size: int = 128
    visualize_every: int = 500
    save_every: int = 1000
    target_update_freq: int = 100  # How often to update target network
    gradient_accumulation_steps: int = 4  # Accumulate gradients for larger effective batch
    max_workers_per_gpu: int = 8  # Limit workers per GPU to avoid memory issues


class SharedReplayBuffer:
    """Thread-safe shared replay buffer for parallel experience collection."""
    
    def __init__(self, maxlen: int = 200000):
        self.maxlen = maxlen
        self.buffer = []
        self.position = 0
        self.lock = threading.Lock()
        
    def push(self, experience):
        """Add experience to buffer in a thread-safe manner."""
        with self.lock:
            if len(self.buffer) < self.maxlen:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience
                self.position = (self.position + 1) % self.maxlen
    
    def sample(self, batch_size: int):
        """Sample batch from buffer."""
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class ParallelEnvironmentWorker:
    """Worker process for running environment simulations."""
    
    def __init__(self, worker_id: int, config: TrainingConfig):
        self.worker_id = worker_id
        self.config = config
        self.env = AdvancedCirclePlacementEnv(map_size=config.map_size)
        
    def run_episode(self, agent_state_dict: Dict, epsilon: float) -> List[Tuple]:
        """Run a single episode and return experiences."""
        # Create temporary agent with shared weights
        temp_agent = GuidedDQNAgent(
            map_size=self.config.map_size,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            buffer_size=1000,  # Small buffer for worker
        )
        temp_agent.q_network.load_state_dict(agent_state_dict)
        temp_agent.epsilon = epsilon
        
        # Generate new map
        weighted_matrix = random_seeder(self.config.map_size, time_steps=100000)
        state = self.env.reset(weighted_matrix)
        
        experiences = []
        episode_reward = 0
        
        while True:
            # Get valid actions
            valid_mask = self.env.get_valid_actions_mask()
            
            # Choose action
            action = temp_agent.act(state, self.env, valid_mask)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            experiences.append((state, action, reward, next_state if not done else None, done))
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        return experiences, episode_reward, info["coverage"]


def worker_process(worker_id: int, config: TrainingConfig, 
                  agent_queue: mp.Queue, experience_queue: mp.Queue, 
                  control_queue: mp.Queue):
    """Worker process function."""
    worker = ParallelEnvironmentWorker(worker_id, config)
    
    while True:
        try:
            # Check for control signals
            try:
                signal = control_queue.get_nowait()
                if signal == "STOP":
                    break
            except queue.Empty:
                pass
            
            # Get latest agent state
            try:
                agent_state_dict, epsilon = agent_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # Run episode
            experiences, reward, coverage = worker.run_episode(agent_state_dict, epsilon)
            
            # Send experiences back
            experience_queue.put((worker_id, experiences, reward, coverage))
            
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            break


class ParallelDQNTrainer:
    """Parallel DQN trainer using multiple processes for environment simulation."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize main agent
        self.agent = GuidedDQNAgent(
            map_size=config.map_size,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
        )
        
        # Shared replay buffer
        self.replay_buffer = SharedReplayBuffer(config.buffer_size)
        
        # Multiprocessing setup
        self.agent_queues = [mp.Queue(maxsize=2) for _ in range(config.n_workers)]
        self.experience_queue = mp.Queue(maxsize=config.n_workers * 2)
        self.control_queues = [mp.Queue() for _ in range(config.n_workers)]
        
        # Start worker processes
        self.workers = []
        for i in range(config.n_workers):
            worker = mp.Process(
                target=worker_process,
                args=(i, config, self.agent_queues[i], self.experience_queue, self.control_queues[i])
            )
            worker.start()
            self.workers.append(worker)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_coverage = []
        self.losses = []
        self.training_step = 0
        
        # Experience collection thread
        self.experience_thread = threading.Thread(target=self._experience_collector, daemon=True)
        self.experience_thread.start()
        
    def _experience_collector(self):
        """Thread to collect experiences from workers."""
        while True:
            try:
                worker_id, experiences, reward, coverage = self.experience_queue.get(timeout=1.0)
                
                # Add experiences to replay buffer
                for exp in experiences:
                    self.replay_buffer.push(exp)
                
                # Record metrics
                self.episode_rewards.append(reward)
                self.episode_coverage.append(coverage)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Experience collector error: {e}")
                break
    
    def _distribute_agent_state(self):
        """Distribute current agent state to all workers."""
        agent_state = self.agent.q_network.state_dict()
        epsilon = self.agent.epsilon
        
        for q in self.agent_queues:
            try:
                # Non-blocking put, skip if queue is full
                q.put_nowait((agent_state, epsilon))
            except queue.Full:
                pass
    
    def _train_step(self):
        """Perform one training step with accumulated gradients."""
        total_loss = 0
        num_batches = 0
        
        # Accumulate gradients over multiple batches
        self.agent.optimizer.zero_grad()
        
        for _ in range(self.config.gradient_accumulation_steps):
            batch = self.replay_buffer.sample(self.config.batch_size)
            if batch is None:
                continue
                
            # Prepare batch data
            states = [e[0] for e in batch]
            actions = [e[1] for e in batch]
            rewards = [e[2] for e in batch]
            next_states = [e[3] for e in batch if e[3] is not None]
            dones = [e[4] for e in batch]
            
            # Forward pass
            state_batch = self.agent._prepare_state_batch(states)
            current_q_values = self.agent.q_network(state_batch)
            
            # Convert actions to indices
            action_indices = torch.LongTensor(
                [[a[0] * self.config.map_size + a[1]] for a in actions]
            ).to(self.device)
            
            current_q_values = (
                current_q_values.view(self.config.batch_size, -1)
                .gather(1, action_indices)
                .squeeze()
            )
            
            # Target Q values
            next_q_values = torch.zeros(self.config.batch_size).to(self.device)
            if next_states:
                next_state_batch = self.agent._prepare_state_batch(next_states)
                non_final_mask = torch.tensor([not d for d in dones], dtype=torch.bool).to(self.device)
                
                with torch.no_grad():
                    # Double DQN
                    next_actions = (
                        self.agent.q_network(next_state_batch)
                        .view(len(next_states), -1)
                        .max(1)[1]
                    )
                    next_q_values[non_final_mask] = (
                        self.agent.target_network(next_state_batch)
                        .view(len(next_states), -1)
                        .gather(1, next_actions.unsqueeze(1))
                        .squeeze()
                    )
            
            # Compute targets and loss
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            targets = rewards_tensor + self.agent.gamma * next_q_values
            loss = nn.functional.smooth_l1_loss(current_q_values, targets)
            
            # Backward pass (accumulate gradients)
            loss.backward()
            total_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            # Apply accumulated gradients
            torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 1.0)
            self.agent.optimizer.step()
            
            # Update target network periodically
            if self.training_step % self.config.target_update_freq == 0:
                for target_param, param in zip(
                    self.agent.target_network.parameters(), 
                    self.agent.q_network.parameters()
                ):
                    target_param.data.copy_(
                        self.agent.tau * param.data + (1.0 - self.agent.tau) * target_param.data
                    )
            
            self.training_step += 1
            return total_loss / num_batches
        
        return None
    
    def train(self):
        """Main training loop."""
        print("=" * 80)
        print(f"PARALLEL DQN TRAINING WITH {self.config.n_workers} WORKERS")
        print("=" * 80)
        print(f"Target episodes: {self.config.n_episodes}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Buffer size: {self.config.buffer_size}")
        print(f"Available CPU cores: {psutil.cpu_count()}")
        print(f"Available memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print("=" * 80)
        
        # Initial distribution of agent state
        self._distribute_agent_state()
        
        pbar = tqdm(total=self.config.n_episodes, desc="Training Episodes")
        last_episode_count = 0
        
        start_time = time.time()
        
        while len(self.episode_rewards) < self.config.n_episodes:
            # Update progress bar
            current_episodes = len(self.episode_rewards)
            pbar.update(current_episodes - last_episode_count)
            last_episode_count = current_episodes
            
            # Perform training step if we have enough experiences
            if len(self.replay_buffer) > self.config.batch_size * self.config.gradient_accumulation_steps:
                loss = self._train_step()
                if loss is not None:
                    self.losses.append(loss)
            
            # Distribute updated agent state periodically
            if self.training_step % 10 == 0:
                self._distribute_agent_state()
            
            # Update progress bar with metrics
            if current_episodes > 0:
                recent_coverage = np.mean(self.episode_coverage[-100:]) if len(self.episode_coverage) >= 100 else np.mean(self.episode_coverage)
                recent_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                
                pbar.set_postfix({
                    "Coverage": f"{recent_coverage:.1%}",
                    "Reward": f"{recent_reward:.2f}",
                    "Buffer": f"{len(self.replay_buffer)}",
                    "Loss": f"{np.mean(self.losses[-100:]):.4f}" if self.losses else "N/A"
                })
            
            # Periodic evaluation and saving
            if current_episodes > 0 and current_episodes % self.config.visualize_every == 0:
                self._evaluate_and_visualize(current_episodes)
            
            if current_episodes > 0 and current_episodes % self.config.save_every == 0:
                self._save_checkpoint(current_episodes)
            
            time.sleep(0.01)  # Small delay to prevent busy waiting
        
        pbar.close()
        
        # Final evaluation
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Episodes per second: {self.config.n_episodes / training_time:.2f}")
        print(f"Final average coverage: {np.mean(self.episode_coverage[-100:]):.1%}")
        print(f"Best coverage achieved: {max(self.episode_coverage):.1%}")
        
        # Save final model
        self._save_checkpoint(self.config.n_episodes, final=True)
        self._plot_training_progress()
        
        # Cleanup
        self._cleanup()
    
    def _evaluate_and_visualize(self, episode: int):
        """Evaluate agent and create visualizations."""
        print(f"\n{'=' * 60}")
        print(f"Episode {episode}/{self.config.n_episodes}")
        print(f"{'=' * 60}")
        
        # Performance stats
        recent_coverage = np.mean(self.episode_coverage[-100:])
        recent_rewards = np.mean(self.episode_rewards[-100:])
        best_coverage = max(self.episode_coverage)
        
        print(f"\nPERFORMANCE:")
        print(f"  Recent avg coverage: {recent_coverage:.1%}")
        print(f"  Best coverage: {best_coverage:.1%}")
        print(f"  Recent avg reward: {recent_rewards:.2f}")
        print(f"  Training steps: {self.training_step}")
        
        if self.losses:
            print(f"\nTRAINING:")
            print(f"  Recent avg loss: {np.mean(self.losses[-100:]):.4f}")
            print(f"  Epsilon: {self.agent.epsilon:.3f}")
        
        # Test on multiple maps in parallel
        print(f"\nTESTING ON 10 RANDOM MAPS:")
        test_coverages = self._parallel_evaluation(n_tests=10)
        print(f"  Test coverages: {[f'{c:.1%}' for c in test_coverages]}")
        print(f"  Average: {np.mean(test_coverages):.1%}")
        print(f"  Std: {np.std(test_coverages):.1%}")
    
    def _parallel_evaluation(self, n_tests: int = 10) -> List[float]:
        """Run parallel evaluation on multiple test environments."""
        def evaluate_single():
            test_env = AdvancedCirclePlacementEnv(self.config.map_size)
            test_map = random_seeder(self.config.map_size, time_steps=100000)
            state = test_env.reset(test_map)
            
            while True:
                valid_mask = test_env.get_valid_actions_mask()
                with torch.no_grad():
                    # Use greedy policy for evaluation
                    old_epsilon = self.agent.epsilon
                    self.agent.epsilon = 0.0
                    action = self.agent.act(state, test_env, valid_mask)
                    self.agent.epsilon = old_epsilon
                    
                state, _, done, info = test_env.step(action)
                if done:
                    return info["coverage"]
        
        # Run evaluations in parallel using thread pool
        with ThreadPoolExecutor(max_workers=min(n_tests, 10)) as executor:
            futures = [executor.submit(evaluate_single) for _ in range(n_tests)]
            coverages = [future.result() for future in futures]
        
        return coverages
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save training checkpoint."""
        suffix = "final" if final else f"ep{episode}"
        filename = f"parallel_rl_model_{suffix}.pth"
        
        torch.save({
            "model_state_dict": self.agent.q_network.state_dict(),
            "target_state_dict": self.agent.target_network.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "episode": episode,
            "training_step": self.training_step,
            "episode_rewards": self.episode_rewards,
            "episode_coverage": self.episode_coverage,
            "losses": self.losses,
            "config": self.config,
        }, filename)
        
        print(f"Model saved to {filename}")
    
    def _plot_training_progress(self):
        """Plot training progress."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Coverage over time
        ax1.plot(self.episode_coverage, alpha=0.3, label="Episode")
        if len(self.episode_coverage) > 100:
            ax1.plot(
                np.convolve(self.episode_coverage, np.ones(100) / 100, mode="valid"),
                label="100-episode average",
                linewidth=2,
            )
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Coverage")
        ax1.set_title("Coverage Over Training")
        ax1.legend()
        ax1.grid(True)
        
        # Rewards over time
        ax2.plot(self.episode_rewards, alpha=0.3, label="Episode")
        if len(self.episode_rewards) > 100:
            ax2.plot(
                np.convolve(self.episode_rewards, np.ones(100) / 100, mode="valid"),
                label="100-episode average",
                linewidth=2,
            )
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Total Reward")
        ax2.set_title("Rewards Over Training")
        ax2.legend()
        ax2.grid(True)
        
        # Loss over time
        if self.losses:
            ax3.plot(self.losses, alpha=0.5)
            if len(self.losses) > 100:
                ax3.plot(
                    np.convolve(self.losses, np.ones(100) / 100, mode="valid"),
                    label="100-step average",
                    linewidth=2,
                )
            ax3.set_xlabel("Training Step")
            ax3.set_ylabel("Loss")
            ax3.set_title("Training Loss")
            ax3.legend()
            ax3.grid(True)
        
        # Coverage distribution
        ax4.hist(self.episode_coverage, bins=50, alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(self.episode_coverage), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.episode_coverage):.1%}')
        ax4.set_xlabel("Coverage")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Coverage Distribution")
        ax4.legend()
        ax4.grid(True)
        
        plt.suptitle(f"Parallel Training Progress - {len(self.episode_rewards)} Episodes")
        plt.tight_layout()
        plt.savefig("parallel_training_progress.png", dpi=150, bbox_inches="tight")
        plt.show()
    
    def _cleanup(self):
        """Cleanup worker processes."""
        print("Cleaning up worker processes...")
        
        # Send stop signal to all workers
        for q in self.control_queues:
            q.put("STOP")
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
        
        print("Cleanup complete.")


def main():
    """Main training function."""
    # Detect optimal number of workers based on system
    n_cores = psutil.cpu_count()
    n_workers = min(64, n_cores - 4)  # Leave some cores for system
    
    print(f"Detected {n_cores} CPU cores, using {n_workers} workers")
    
    config = TrainingConfig(
        n_episodes=50000,
        n_workers=n_workers,
        batch_size=128,  # Larger batch for better GPU utilization
        buffer_size=500000,  # Large buffer for diverse experiences
        visualize_every=1000,
        save_every=2000,
    )
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    trainer = ParallelDQNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()