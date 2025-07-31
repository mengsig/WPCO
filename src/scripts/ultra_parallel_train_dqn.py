import sys
import os
import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import psutil
from multiprocessing import shared_memory
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.algorithms.dqn_agent import (
    AdvancedCirclePlacementEnv,
    GuidedDQNAgent,
    random_seeder,
)


@dataclass
class UltraTrainingConfig:
    """Configuration for ultra-parallel training."""
    n_episodes: int = 100000
    n_workers: int = 120  # Use almost all cores
    batch_size: int = 256  # Very large batch size
    buffer_size: int = 1000000  # Massive buffer
    learning_rate: float = 2e-4  # Slightly higher for larger batches
    map_size: int = 128
    visualize_every: int = 2000
    save_every: int = 5000
    target_update_freq: int = 50  # More frequent updates
    gradient_accumulation_steps: int = 8  # Large accumulation
    experience_chunk_size: int = 1000  # Process experiences in chunks
    async_training: bool = True  # Asynchronous training updates
    shared_memory_size: int = 50000000  # 50MB shared memory buffer


class SharedMemoryBuffer:
    """Ultra-fast shared memory buffer for experience replay."""
    
    def __init__(self, max_size: int, experience_size: int = 1000):
        self.max_size = max_size
        self.experience_size = experience_size
        
        # Create shared memory for experiences
        self.shm_size = max_size * experience_size
        self.shm = shared_memory.SharedMemory(create=True, size=self.shm_size)
        self.buffer = np.ndarray((max_size,), dtype=f'S{experience_size}', buffer=self.shm.buf)
        
        # Shared counters
        self.count_shm = shared_memory.SharedMemory(create=True, size=8)
        self.count = np.ndarray((1,), dtype=np.int64, buffer=self.count_shm.buf)
        self.count[0] = 0
        
        self.position_shm = shared_memory.SharedMemory(create=True, size=8)
        self.position = np.ndarray((1,), dtype=np.int64, buffer=self.position_shm.buf)
        self.position[0] = 0
        
        # Lock for thread safety
        self.lock = mp.Lock()
    
    def push(self, experience):
        """Add experience to shared buffer."""
        serialized = pickle.dumps(experience)
        if len(serialized) > self.experience_size:
            return  # Skip if too large
            
        with self.lock:
            self.buffer[self.position[0]] = serialized
            self.position[0] = (self.position[0] + 1) % self.max_size
            if self.count[0] < self.max_size:
                self.count[0] += 1
    
    def sample(self, batch_size: int):
        """Sample batch from shared buffer."""
        with self.lock:
            if self.count[0] < batch_size:
                return None
            
            indices = np.random.choice(self.count[0], batch_size, replace=False)
            experiences = []
            
            for idx in indices:
                try:
                    exp = pickle.loads(self.buffer[idx])
                    experiences.append(exp)
                except:
                    continue
            
            return experiences if len(experiences) == batch_size else None
    
    def __len__(self):
        return self.count[0]
    
    def cleanup(self):
        """Cleanup shared memory."""
        self.shm.close()
        self.shm.unlink()
        self.count_shm.close()
        self.count_shm.unlink()
        self.position_shm.close()
        self.position_shm.unlink()


class VectorizedEnvironmentWorker:
    """Highly optimized worker that can run multiple episodes in parallel."""
    
    def __init__(self, worker_id: int, config: UltraTrainingConfig, n_envs: int = 4):
        self.worker_id = worker_id
        self.config = config
        self.n_envs = n_envs
        
        # Create multiple environments for vectorized execution
        self.envs = [AdvancedCirclePlacementEnv(map_size=config.map_size) for _ in range(n_envs)]
        
    def run_vectorized_episodes(self, agent_state_dict: Dict, epsilon: float) -> List[Tuple]:
        """Run multiple episodes in parallel and return all experiences."""
        # Create temporary agent
        temp_agent = GuidedDQNAgent(
            map_size=self.config.map_size,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            buffer_size=1000,
        )
        temp_agent.q_network.load_state_dict(agent_state_dict)
        temp_agent.epsilon = epsilon
        
        all_experiences = []
        episode_rewards = []
        episode_coverages = []
        
        # Run episodes in parallel
        for env in self.envs:
            # Generate new map
            weighted_matrix = random_seeder(self.config.map_size, time_steps=100000)
            state = env.reset(weighted_matrix)
            
            experiences = []
            episode_reward = 0
            
            while True:
                # Get valid actions
                valid_mask = env.get_valid_actions_mask()
                
                # Choose action
                action = temp_agent.act(state, env, valid_mask)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                experiences.append((state, action, reward, next_state if not done else None, done))
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            all_experiences.extend(experiences)
            episode_rewards.append(episode_reward)
            episode_coverages.append(info["coverage"])
        
        return all_experiences, episode_rewards, episode_coverages


def ultra_worker_process(worker_id: int, config: UltraTrainingConfig,
                        agent_queue: mp.Queue, experience_queue: mp.Queue,
                        control_queue: mp.Queue, shm_name: str):
    """Ultra-optimized worker process."""
    # Create vectorized worker
    n_envs_per_worker = max(1, 8 // config.n_workers) if config.n_workers > 8 else 2
    worker = VectorizedEnvironmentWorker(worker_id, config, n_envs_per_worker)
    
    episodes_completed = 0
    
    while True:
        try:
            # Check for control signals
            try:
                signal = control_queue.get_nowait()
                if signal == "STOP":
                    break
            except:
                pass
            
            # Get latest agent state
            try:
                agent_state_dict, epsilon = agent_queue.get(timeout=0.1)
            except:
                continue
            
            # Run vectorized episodes
            experiences, rewards, coverages = worker.run_vectorized_episodes(agent_state_dict, epsilon)
            
            # Send results back in chunks for better performance
            chunk_size = config.experience_chunk_size
            for i in range(0, len(experiences), chunk_size):
                chunk = experiences[i:i + chunk_size]
                experience_queue.put((worker_id, chunk, rewards, coverages))
            
            episodes_completed += len(rewards)
            
        except Exception as e:
            print(f"Ultra worker {worker_id} error: {e}")
            break


class UltraParallelDQNTrainer:
    """Ultra-high performance parallel DQN trainer."""
    
    def __init__(self, config: UltraTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Initialize main agent with optimizations
        self.agent = GuidedDQNAgent(
            map_size=config.map_size,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
        )
        
        # Enable mixed precision training if available
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Ultra-fast shared memory buffer
        try:
            self.replay_buffer = SharedMemoryBuffer(config.buffer_size)
        except:
            print("Falling back to regular buffer due to shared memory limitations")
            from parallel_train_dqn import SharedReplayBuffer
            self.replay_buffer = SharedReplayBuffer(config.buffer_size)
        
        # Multiprocessing setup
        self.agent_queues = [mp.Queue(maxsize=1) for _ in range(config.n_workers)]
        self.experience_queue = mp.Queue(maxsize=config.n_workers * 4)
        self.control_queues = [mp.Queue() for _ in range(config.n_workers)]
        
        # Start ultra workers
        self.workers = []
        for i in range(config.n_workers):
            worker = mp.Process(
                target=ultra_worker_process,
                args=(i, config, self.agent_queues[i], self.experience_queue, 
                     self.control_queues[i], getattr(self.replay_buffer, 'shm_name', None))
            )
            worker.start()
            self.workers.append(worker)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_coverage = []
        self.losses = []
        self.training_step = 0
        self.last_distribution_time = 0
        
        # High-performance experience collection
        self.experience_threads = []
        for _ in range(min(4, config.n_workers // 16)):  # Multiple collector threads
            thread = threading.Thread(target=self._ultra_experience_collector, daemon=True)
            thread.start()
            self.experience_threads.append(thread)
        
        # Asynchronous training thread
        if config.async_training:
            self.training_thread = threading.Thread(target=self._async_training_loop, daemon=True)
            self.training_thread.start()
    
    def _ultra_experience_collector(self):
        """Ultra-fast experience collection with batching."""
        batch_experiences = []
        batch_rewards = []
        batch_coverages = []
        
        while True:
            try:
                worker_id, experiences, rewards, coverages = self.experience_queue.get(timeout=1.0)
                
                # Batch experiences for efficiency
                batch_experiences.extend(experiences)
                batch_rewards.extend(rewards)
                batch_coverages.extend(coverages)
                
                # Process in large batches
                if len(batch_experiences) >= self.config.experience_chunk_size:
                    # Add to replay buffer
                    for exp in batch_experiences:
                        self.replay_buffer.push(exp)
                    
                    # Record metrics
                    self.episode_rewards.extend(batch_rewards)
                    self.episode_coverage.extend(batch_coverages)
                    
                    # Clear batches
                    batch_experiences.clear()
                    batch_rewards.clear()
                    batch_coverages.clear()
                
            except:
                continue
    
    def _async_training_loop(self):
        """Asynchronous training loop for continuous learning."""
        while True:
            try:
                if len(self.replay_buffer) > self.config.batch_size * self.config.gradient_accumulation_steps:
                    loss = self._ultra_train_step()
                    if loss is not None:
                        self.losses.append(loss)
                
                time.sleep(0.001)  # Very small delay
                
            except Exception as e:
                print(f"Async training error: {e}")
                time.sleep(0.1)
    
    def _distribute_agent_state(self):
        """Ultra-fast agent state distribution."""
        current_time = time.time()
        if current_time - self.last_distribution_time < 0.1:  # Rate limit
            return
        
        agent_state = self.agent.q_network.state_dict()
        epsilon = self.agent.epsilon
        
        # Distribute to all workers non-blocking
        for q in self.agent_queues:
            try:
                if q.empty():  # Only send if queue is empty
                    q.put_nowait((agent_state, epsilon))
            except:
                pass
        
        self.last_distribution_time = current_time
    
    def _ultra_train_step(self):
        """Ultra-optimized training step with mixed precision."""
        total_loss = 0
        num_batches = 0
        
        self.agent.optimizer.zero_grad()
        
        # Use mixed precision if available
        with torch.cuda.amp.autocast() if self.scaler else torch.no_grad():
            for _ in range(self.config.gradient_accumulation_steps):
                batch = self.replay_buffer.sample(self.config.batch_size)
                if batch is None:
                    continue
                
                try:
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
                    
                    # Target Q values with double DQN
                    next_q_values = torch.zeros(self.config.batch_size).to(self.device)
                    if next_states:
                        next_state_batch = self.agent._prepare_state_batch(next_states)
                        non_final_mask = torch.tensor([not d for d in dones], dtype=torch.bool).to(self.device)
                        
                        with torch.no_grad():
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
                    
                    # Compute loss
                    rewards_tensor = torch.FloatTensor(rewards).to(self.device)
                    targets = rewards_tensor + self.agent.gamma * next_q_values
                    loss = nn.functional.smooth_l1_loss(current_q_values, targets)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Training step error: {e}")
                    continue
        
        if num_batches > 0:
            # Apply gradients
            if self.scaler:
                self.scaler.unscale_(self.agent.optimizer)
                torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 1.0)
                self.scaler.step(self.agent.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 1.0)
                self.agent.optimizer.step()
            
            # Update target network
            if self.training_step % self.config.target_update_freq == 0:
                for target_param, param in zip(
                    self.agent.target_network.parameters(),
                    self.agent.q_network.parameters()
                ):
                    target_param.data.copy_(
                        self.agent.tau * param.data + (1.0 - self.agent.tau) * target_param.data
                    )
            
            self.training_step += 1
            return total_loss * self.config.gradient_accumulation_steps
        
        return None
    
    def train(self):
        """Ultra-high performance training loop."""
        print("=" * 100)
        print(f"ULTRA-PARALLEL DQN TRAINING WITH {self.config.n_workers} WORKERS")
        print("=" * 100)
        print(f"Target episodes: {self.config.n_episodes:,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Buffer size: {self.config.buffer_size:,}")
        print(f"Available CPU cores: {psutil.cpu_count()}")
        print(f"Available memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"Mixed precision: {self.scaler is not None}")
        print(f"Async training: {self.config.async_training}")
        print("=" * 100)
        
        # Initial distribution
        self._distribute_agent_state()
        
        pbar = tqdm(total=self.config.n_episodes, desc="Ultra Training")
        last_episode_count = 0
        start_time = time.time()
        last_time = start_time
        
        while len(self.episode_rewards) < self.config.n_episodes:
            current_episodes = len(self.episode_rewards)
            pbar.update(current_episodes - last_episode_count)
            last_episode_count = current_episodes
            
            # Non-async training
            if not self.config.async_training:
                if len(self.replay_buffer) > self.config.batch_size * self.config.gradient_accumulation_steps:
                    loss = self._ultra_train_step()
                    if loss is not None:
                        self.losses.append(loss)
            
            # Distribute agent state
            self._distribute_agent_state()
            
            # Update progress bar
            if current_episodes > 0:
                current_time = time.time()
                episodes_per_sec = (current_episodes - (last_episode_count - (current_episodes - last_episode_count))) / max(current_time - last_time, 0.1)
                last_time = current_time
                
                recent_coverage = np.mean(self.episode_coverage[-1000:]) if len(self.episode_coverage) >= 1000 else np.mean(self.episode_coverage) if self.episode_coverage else 0
                recent_reward = np.mean(self.episode_rewards[-1000:]) if len(self.episode_rewards) >= 1000 else np.mean(self.episode_rewards) if self.episode_rewards else 0
                
                pbar.set_postfix({
                    "Coverage": f"{recent_coverage:.1%}",
                    "Reward": f"{recent_reward:.2f}",
                    "Buffer": f"{len(self.replay_buffer):,}",
                    "Loss": f"{np.mean(self.losses[-100:]):.4f}" if self.losses else "N/A",
                    "Eps/s": f"{episodes_per_sec:.1f}"
                })
            
            # Periodic evaluation
            if current_episodes > 0 and current_episodes % self.config.visualize_every == 0:
                self._evaluate_and_save(current_episodes)
            
            time.sleep(0.001)  # Minimal delay
        
        pbar.close()
        
        # Final statistics
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n" + "=" * 100)
        print("ULTRA TRAINING COMPLETE!")
        print("=" * 100)
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        print(f"Episodes per second: {self.config.n_episodes / training_time:.2f}")
        print(f"Final average coverage: {np.mean(self.episode_coverage[-1000:]):.1%}")
        print(f"Best coverage achieved: {max(self.episode_coverage):.1%}")
        print(f"Total training steps: {self.training_step:,}")
        
        self._save_final_model()
        self._cleanup()
    
    def _evaluate_and_save(self, episode: int):
        """Fast evaluation and saving."""
        print(f"\nEpisode {episode:,}/{self.config.n_episodes:,}")
        print(f"Recent coverage: {np.mean(self.episode_coverage[-1000:]):.1%}")
        print(f"Best coverage: {max(self.episode_coverage):.1%}")
        print(f"Training steps: {self.training_step:,}")
        print(f"Buffer size: {len(self.replay_buffer):,}")
        
        # Save checkpoint
        if episode % self.config.save_every == 0:
            torch.save({
                "model_state_dict": self.agent.q_network.state_dict(),
                "episode": episode,
                "training_step": self.training_step,
                "coverage_stats": {
                    "recent": np.mean(self.episode_coverage[-1000:]),
                    "best": max(self.episode_coverage),
                    "std": np.std(self.episode_coverage[-1000:])
                }
            }, f"ultra_model_ep{episode}.pth")
    
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
        }, "ultra_final_model.pth")
        
        # Plot results
        self._plot_results()
    
    def _plot_results(self):
        """Plot training results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Coverage
        ax1.plot(self.episode_coverage, alpha=0.3)
        if len(self.episode_coverage) > 1000:
            smoothed = np.convolve(self.episode_coverage, np.ones(1000)/1000, mode='valid')
            ax1.plot(smoothed, linewidth=2, label='1000-episode average')
        ax1.set_title('Coverage Over Time')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Coverage')
        ax1.legend()
        ax1.grid(True)
        
        # Rewards
        ax2.plot(self.episode_rewards, alpha=0.3)
        if len(self.episode_rewards) > 1000:
            smoothed = np.convolve(self.episode_rewards, np.ones(1000)/1000, mode='valid')
            ax2.plot(smoothed, linewidth=2, label='1000-episode average')
        ax2.set_title('Rewards Over Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True)
        
        # Loss
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
        
        plt.suptitle(f'Ultra-Parallel Training Results - {len(self.episode_rewards):,} Episodes')
        plt.tight_layout()
        plt.savefig('ultra_training_results.png', dpi=150, bbox_inches='tight')
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
        
        # Cleanup shared memory
        if hasattr(self.replay_buffer, 'cleanup'):
            try:
                self.replay_buffer.cleanup()
            except:
                pass
        
        print("Cleanup complete.")


def main():
    """Main ultra training function."""
    # System detection
    n_cores = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Optimize worker count based on system
    n_workers = min(120, max(64, n_cores - 8))  # Leave cores for system
    
    print(f"System: {n_cores} cores, {memory_gb:.1f} GB RAM")
    print(f"Using {n_workers} workers for ultra-parallel training")
    
    config = UltraTrainingConfig(
        n_episodes=100000,
        n_workers=n_workers,
        batch_size=256,
        buffer_size=1000000,
        learning_rate=2e-4,
        visualize_every=2000,
        save_every=5000,
        async_training=True,
    )
    
    # Set optimal multiprocessing method
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass
    
    trainer = UltraParallelDQNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()