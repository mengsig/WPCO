#!/usr/bin/env python3
"""Asynchronous enhanced training script with multiple workers."""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import random
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.algorithms.dqn_agent import (
    AdvancedCirclePlacementEnv,
    GuidedDQNAgent,
    random_seeder,
    compute_included,
)
from src.algorithms.advanced_enhancements import (
    AdvancedFeatureExtractor,
    ImprovedCirclePlacementNet,
    CurriculumLearningScheduler,
    ExperiencePrioritization,
    create_enhanced_training_config,
)
from src.utils.periodic_tracker import RobustPeriodicChecker
from src.scripts.enhanced_train import EnhancedCirclePlacementEnv, EnhancedDQNAgent


@dataclass
class EnhancedExperience:
    """Enhanced experience with additional metadata."""
    state: Dict
    action: Tuple[int, int]
    reward: float
    next_state: Optional[Dict]
    done: bool
    coverage: float
    coverage_improvement: float
    td_error: float = 0.0
    worker_id: int = 0


class ThreadSafeEnhancedReplayBuffer:
    """Thread-safe replay buffer for enhanced experiences."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.prioritizer = ExperiencePrioritization()
    
    def push(self, experience: EnhancedExperience):
        """Add experience to buffer."""
        with self.lock:
            self.buffer.append(experience)
            # Add to prioritizer
            self.prioritizer.add_experience(
                experience.td_error, 
                experience.coverage_improvement
            )
    
    def sample(self, batch_size: int) -> List[EnhancedExperience]:
        """Sample batch of experiences."""
        with self.lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            
            # Use prioritized sampling
            indices = self.prioritizer.sample_indices(batch_size, len(self.buffer))
            return [self.buffer[i] for i in indices]
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities for sampled experiences."""
        with self.lock:
            self.prioritizer.update_priorities(indices, td_errors)
    
    def __len__(self):
        with self.lock:
            return len(self.buffer)


def enhanced_worker_process(
    config: Dict,
    result_queue: mp.Queue,
    model_queue: mp.Queue,
    epsilon_value: mp.Value,
    worker_id: int,
    curriculum_difficulty: mp.Value
):
    """Worker process for collecting enhanced experiences."""
    print(f"Worker {worker_id} starting...")
    
    # Create environment with advanced features
    env = EnhancedCirclePlacementEnv(
        map_size=config['map_size'],
        use_advanced_features=config['use_advanced_features']
    )
    
    # Create local agent (will sync weights periodically)
    local_agent = EnhancedDQNAgent(map_size=config['map_size'], config=config)
    local_agent.epsilon = epsilon_value.value
    
    episodes_processed = 0
    last_model_update = time.time()
    
    while True:
        try:
            # Update model weights periodically
            if time.time() - last_model_update > 10:  # Every 10 seconds
                try:
                    if not model_queue.empty():
                        model_weights = model_queue.get_nowait()
                        local_agent.q_network.load_state_dict(model_weights)
                        last_model_update = time.time()
                except:
                    pass
            
            # Adjust difficulty based on curriculum
            if config.get('use_curriculum', False):
                difficulty = curriculum_difficulty.value
                base_radii = [20, 17, 14, 12, 12, 8, 7, 6, 5, 4, 3, 2, 1]
                n_circles = int(len(base_radii) * difficulty)
                n_circles = max(3, n_circles)
                env.radii = base_radii[:n_circles]
            
            # Generate random map
            weighted_matrix = random_seeder(config['map_size'], time_steps=100000)
            state = env.reset(weighted_matrix)
            
            episode_experiences = []
            episode_reward = 0
            previous_coverage = 0
            
            # Update epsilon
            local_agent.epsilon = epsilon_value.value
            
            while state is not None:
                # Choose action with enhanced suggestions
                if hasattr(env, 'get_placement_suggestions'):
                    action = local_agent.act_with_suggestions(state, env)
                else:
                    action = local_agent.act(state, env)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Calculate coverage improvement
                coverage_improvement = info['coverage'] - previous_coverage
                
                # Add coverage improvement bonus
                if coverage_improvement > 0:
                    reward += config.get('coverage_bonus_weight', 0.5) * coverage_improvement
                
                # Add exploration bonus
                if len(env.placed_circles) > 3:
                    reward += config.get('exploration_bonus', 0.1) * np.random.random() * local_agent.epsilon
                
                # Create enhanced experience
                experience = EnhancedExperience(
                    state=state,
                    action=action,
                    reward=reward * config.get('reward_scaling', 1.0),
                    next_state=next_state,
                    done=done,
                    coverage=info['coverage'],
                    coverage_improvement=coverage_improvement,
                    worker_id=worker_id
                )
                
                episode_experiences.append(experience)
                episode_reward += reward
                previous_coverage = info['coverage']
                state = next_state
                
                if done:
                    break
            
            # Send episode results
            result = {
                'experiences': episode_experiences,
                'episode_reward': episode_reward,
                'coverage': info['coverage'],
                'n_circles': len(env.placed_circles),
                'worker_id': worker_id,
                'radii_config': env.radii
            }
            
            result_queue.put(result)
            episodes_processed += 1
            
            if episodes_processed % 100 == 0:
                print(f"Worker {worker_id}: {episodes_processed} episodes processed")
                
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)


class AsyncEnhancedTrainer:
    """Asynchronous trainer with enhanced features."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize enhanced agent
        self.agent = EnhancedDQNAgent(
            map_size=config['map_size'],
            config=config
        )
        
        # Replay buffer
        self.replay_buffer = ThreadSafeEnhancedReplayBuffer(config['buffer_size'])
        
        # Shared values
        self.epsilon_value = mp.Value('f', config['epsilon_start'])
        self.curriculum_difficulty = mp.Value('f', 0.3)
        
        # Queues
        self.result_queue = mp.Queue(maxsize=config['n_workers'] * 2)
        self.model_queue = mp.Queue(maxsize=config['n_workers'])
        
        # Curriculum learning
        if config.get('use_curriculum', False):
            self.curriculum = CurriculumLearningScheduler(
                warmup_episodes=config.get('warmup_episodes', 10000)
            )
        else:
            self.curriculum = None
        
        # Metrics
        self.episode_rewards = []
        self.episode_coverage = []
        self.losses = []
        self.training_step = 0
        self.episodes_completed = 0
        
        # Periodic checkers
        self.save_checker = RobustPeriodicChecker(config.get('save_freq', 1000))
        self.eval_checker = RobustPeriodicChecker(config.get('eval_freq', 500))
        self.model_broadcast_checker = RobustPeriodicChecker(100)  # Broadcast model every 100 episodes
        
        # Start workers
        self.workers = []
        for i in range(config['n_workers']):
            worker = mp.Process(
                target=enhanced_worker_process,
                args=(
                    config,
                    self.result_queue,
                    self.model_queue,
                    self.epsilon_value,
                    i,
                    self.curriculum_difficulty
                ),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        # Start experience collection thread
        self.collection_thread = threading.Thread(
            target=self._collect_experiences,
            daemon=True
        )
        self.collection_thread.start()
    
    def _collect_experiences(self):
        """Collect experiences from workers."""
        while True:
            try:
                result = self.result_queue.get(timeout=1)
                
                # Add experiences to replay buffer
                for exp in result['experiences']:
                    self.replay_buffer.push(exp)
                
                # Update metrics
                self.episode_rewards.append(result['episode_reward'])
                self.episode_coverage.append(result['coverage'])
                self.episodes_completed += 1
                
                # Update curriculum
                if self.curriculum:
                    self.curriculum.step()
                    self.curriculum_difficulty.value = self.curriculum.get_difficulty()
                
                # Update epsilon
                self.epsilon_value.value = self.agent.epsilon_end + (
                    self.agent.epsilon_start - self.agent.epsilon_end
                ) * np.exp(-1.0 * self.episodes_completed / self.agent.epsilon_decay_steps)
                
                # Broadcast model periodically
                if self.model_broadcast_checker.should_execute(self.episodes_completed):
                    self._broadcast_model()
                    
            except:
                continue
    
    def _broadcast_model(self):
        """Broadcast current model to workers."""
        model_state = self.agent.q_network.state_dict()
        for _ in range(self.config['n_workers']):
            try:
                self.model_queue.put_nowait(model_state)
            except:
                pass
    
    def _training_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.config['batch_size']:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config['batch_size'])
        
        # Convert to agent's memory format for compatibility
        self.agent.memory.clear()
        for exp in batch:
            self.agent.memory.append((
                exp.state,
                exp.action,
                exp.reward,
                exp.next_state,
                exp.done
            ))
        
        # Train using enhanced replay
        loss = self.agent.enhanced_replay()
        
        if loss is not None:
            self.losses.append(loss)
            self.training_step += 1
        
        return loss
    
    def train(self, total_episodes: int):
        """Main training loop."""
        print("=" * 80)
        print("ASYNCHRONOUS ENHANCED TRAINING")
        print(f"Workers: {self.config['n_workers']}")
        print(f"Device: {self.device}")
        print("=" * 80)
        
        pbar = tqdm(total=total_episodes, desc="Training")
        best_coverage = 0
        
        while self.episodes_completed < total_episodes:
            # Perform training step
            loss = self._training_step()
            
            # Update progress bar
            if self.episodes_completed > 0:
                recent_coverage = np.mean(self.episode_coverage[-100:])
                recent_reward = np.mean(self.episode_rewards[-100:])
                
                # Update best coverage
                if len(self.episode_coverage) > 0:
                    current_best = max(self.episode_coverage[-100:])
                    if current_best > best_coverage:
                        best_coverage = current_best
                
                pbar.n = self.episodes_completed
                pbar.set_postfix({
                    'Coverage': f'{recent_coverage:.1%}',
                    'Best': f'{best_coverage:.1%}',
                    'Reward': f'{recent_reward:.1f}',
                    'Buffer': f'{len(self.replay_buffer):,}',
                    'Loss': f'{np.mean(self.losses[-100:]):.4f}' if self.losses else 'N/A',
                    'Epsilon': f'{self.epsilon_value.value:.3f}'
                })
                pbar.refresh()
            
            # Periodic evaluation
            if self.eval_checker.should_execute(self.episodes_completed):
                print(f"\n[Episode {self.episodes_completed}] Coverage: {recent_coverage:.1%} (Best: {best_coverage:.1%})")
                if self.curriculum:
                    print(f"Curriculum difficulty: {self.curriculum_difficulty.value:.2f}")
            
            # Save checkpoint
            if self.save_checker.should_execute(self.episodes_completed):
                self._save_checkpoint(best_coverage)
            
            time.sleep(0.001)  # Small delay to prevent CPU spinning
        
        pbar.close()
        
        # Cleanup
        for worker in self.workers:
            worker.terminate()
        
        print(f"\nTraining complete! Best coverage: {best_coverage:.1%}")
        return self.agent, best_coverage
    
    def _save_checkpoint(self, best_coverage: float):
        """Save training checkpoint."""
        checkpoint = {
            'episode': self.episodes_completed,
            'model_state': self.agent.q_network.state_dict(),
            'optimizer_state': self.agent.optimizer.state_dict(),
            'best_coverage': best_coverage,
            'config': self.config,
            'episode_rewards': self.episode_rewards[-1000:],
            'episode_coverage': self.episode_coverage[-1000:]
        }
        
        filename = f'enhanced_async_ep{self.episodes_completed}_cov{best_coverage:.0%}.pth'
        torch.save(checkpoint, filename)
        print(f"\nSaved checkpoint: {filename}")


def create_async_config():
    """Create configuration for asynchronous enhanced training."""
    base_config = create_enhanced_training_config()
    
    # Add async-specific settings
    async_config = {
        **base_config,
        'n_workers': mp.cpu_count() - 1,  # Leave one CPU for main thread
        'map_size': 128,
        'episodes': 100000,
        'buffer_size': 500000,  # Larger buffer for async
        'batch_size': 128,  # Larger batch for efficiency
        'save_freq': 2000,
        'eval_freq': 1000,
    }
    
    return async_config


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Create configuration
    config = create_async_config()
    
    # Override with command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50000)
    parser.add_argument('--workers', type=int, default=config['n_workers'])
    args = parser.parse_args()
    
    config['episodes'] = args.episodes
    config['n_workers'] = args.workers
    
    # Create trainer and run
    trainer = AsyncEnhancedTrainer(config)
    agent, best_coverage = trainer.train(config['episodes'])
    
    print(f"\nFinal best coverage: {best_coverage:.1%}")