#!/usr/bin/env python3

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
        if not hasattr(self, "feature_extractor"):
            self.feature_extractor = HeatmapFeatureExtractor(self.map_size)

        self.previous_coverage = 0.0
        return self._get_enhanced_state()

    def step(self, action):
        """Enhanced step with better reward function."""
        x, y = action
        radius = self.radii[self.current_radius_idx]

        # Store current coverage before action
        current_coverage = 1 - (
            self.current_map.sum() / (self.original_map.sum() + 1e-8)
        )

        # Calculate base collection value
        included_weight = compute_included(self.current_map, x, y, radius)

        # Enhanced overlap detection with gradual penalty
        overlap_penalty = 0.0
        touching_bonus = 0.0
        for px, py, pr in self.placed_circles:
            distance = np.sqrt((x - px) ** 2 + (y - py) ** 2)
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

        # IMPORTANT: Normalize and clip reward to prevent Q-value explosion
        # This helps stabilize training and prevent increasing loss
        reward = reward / 100.0  # Scale down large rewards
        reward = np.clip(reward, -10.0, 10.0)  # Clip to reasonable range

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
                valid_positions = np.argwhere(
                    valid_mask.reshape(self.map_size, self.map_size)
                )
                if len(valid_positions) > 0:
                    # Calculate local values for valid positions
                    values = []
                    sample_size = min(100, len(valid_positions))
                    sampled_positions = valid_positions[
                        np.random.choice(
                            len(valid_positions), sample_size, replace=False
                        )
                    ]

                    for pos in sampled_positions:
                        local_value = self._calculate_local_value(
                            current_map, pos[0], pos[1], current_radius
                        )
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
                np.random.randint(current_radius, self.map_size - current_radius),
            )
        else:
            # Enhanced greedy strategy
            best_value = -1
            best_action = None

            # Strategy 1: Try positions near existing circles for tight packing
            if placed_circles and np.random.random() < 0.3:
                for px, py, pr in placed_circles[-3:]:  # Check last 3 circles
                    # Try positions around this circle
                    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
                    for angle in angles:
                        # Position at touching distance
                        dist = pr + current_radius + 1
                        x = int(px + dist * np.cos(angle))
                        y = int(py + dist * np.sin(angle))

                        # Check bounds
                        if (
                            current_radius <= x < self.map_size - current_radius
                            and current_radius <= y < self.map_size - current_radius
                        ):
                            local_value = self._calculate_local_value(
                                current_map, x, y, current_radius
                            )
                            if local_value > best_value:
                                best_value = local_value
                                best_action = (x, y)

            # Strategy 2: Find high-value clusters
            high_value_threshold = (
                np.percentile(current_map[current_map > 0], 75)
                if current_map.max() > 0
                else 0
            )
            high_value_positions = np.argwhere(current_map > high_value_threshold)

            if len(high_value_positions) > 0:
                # Sample positions around high-value areas
                n_samples = min(20, len(high_value_positions))
                sampled_positions = high_value_positions[
                    np.random.choice(
                        len(high_value_positions), n_samples, replace=False
                    )
                ]

                for center in sampled_positions:
                    # Try the center and nearby positions
                    for dx in [-current_radius // 2, 0, current_radius // 2]:
                        for dy in [-current_radius // 2, 0, current_radius // 2]:
                            x = np.clip(
                                center[0] + dx,
                                current_radius,
                                self.map_size - current_radius - 1,
                            )
                            y = np.clip(
                                center[1] + dy,
                                current_radius,
                                self.map_size - current_radius - 1,
                            )

                            local_value = self._calculate_local_value(
                                current_map, x, y, current_radius
                            )
                            if local_value > best_value:
                                best_value = local_value
                                best_action = (x, y)

            if best_action is not None:
                return best_action

            # Fallback
            return (
                np.random.randint(current_radius, self.map_size - current_radius),
                np.random.randint(current_radius, self.map_size - current_radius),
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
            efficiency_scores = []  # Track efficiency scores

            # Store radii configuration for this episode
            radii_config = env.radii.copy()
            n_circles = len(radii_config)

            while True:
                # Get valid actions mask
                current_radius = (
                    env.radii[env.current_radius_idx]
                    if env.current_radius_idx < len(env.radii)
                    else env.radii[-1]
                )
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
                action = agent.act(
                    state_with_info, valid_mask.flatten(), current_epsilon
                )

                # Take step
                next_state, reward, done, info = env.step(action)

                # Store experience
                experience = (
                    state,
                    action,
                    reward,
                    next_state if not done else None,
                    done,
                )
                episode_experiences.append(experience)

                episode_reward += reward
                total_coverage_improvement += info.get("coverage_improvement", 0)

                # Track efficiency
                if "efficiency" in info:
                    efficiency_scores.append(info["efficiency"])

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
                "avg_coverage_improvement": total_coverage_improvement
                / max(len(episode_experiences), 1),
                "worker_id": worker_id,
                "episodes_completed": episodes_completed,
                "maps_generated": maps_generated,
                "radii_config": radii_config,
                "n_circles": n_circles,
                "map_diversity_stats": {
                    "map_mean": env.original_map.mean(),
                    "map_std": env.original_map.std(),
                    "map_max": env.original_map.max(),
                },
                "efficiency_score": np.mean(efficiency_scores)
                if efficiency_scores
                else 0.0,
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
            available_buffer = [
                exp for exp in self.buffer[: len(self.buffer)] if exp is not None
            ]
            if len(available_buffer) < batch_size:
                return None
            indices = np.random.choice(len(available_buffer), batch_size, replace=False)
            return [available_buffer[i] for i in indices]

    def __len__(self):
        with self.lock:
            return len([exp for exp in self.buffer if exp is not None])


@dataclass
class EnhancedRandomizedRadiiConfig:
    """Configuration for enhanced randomized radii training."""

    n_episodes: int = 20000  # 2 million episodes for massive simulation
    n_workers: int = 64  # Increased from 32 to use more cores
    map_size: int = 128
    batch_size: int = 64  # Increased from 128
    buffer_size: int = 1000000  # Increased from 500000
    gradient_accumulation_steps: int = 4  # Increased from 2
    learning_rate: float = 5e-5  # Reduced from 1e-4
    epsilon_start: float = 1.0  # Start with full exploration
    epsilon_end: float = 0.01  # Very low final exploration
    epsilon_decay_episodes: int = 15000  # Decay over 75% of training
    target_update_freq: int = 2000  # Increased from 1000
    visualize_every: int = 1000  # Save image every 5000 episodes
    checkpoint_every: int = 5000  # Backup model every 100k episodes
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
            epsilon_decay_steps=999999999,  # Set to huge number to disable internal decay
            batch_size=config.batch_size,
            tau=0.005,  # Slightly higher for faster target updates
        )

        # Override agent's epsilon with our controlled value
        self.agent.epsilon = config.epsilon_start

        # Use mixed precision if available
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")

        # Shared epsilon value
        self.epsilon_value = mp.Value("f", config.epsilon_start)

        print(f"\n🎯 Initial epsilon configuration:")
        print(f"   Config epsilon_start: {config.epsilon_start}")
        print(f"   Agent epsilon: {self.agent.epsilon}")
        print(f"   Shared epsilon value: {self.epsilon_value.value}")

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
                daemon=True,
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

        # Fixed save frequencies for massive simulation
        # Images: every 5000 episodes (400 total images over 2M episodes)
        # Models: every 100k episodes (20 model backups)
        # Progress: every 10k episodes (200 updates)

        self.model_save_checker = RobustPeriodicChecker(self.config.checkpoint_every)
        self.image_save_checker = RobustPeriodicChecker(self.config.visualize_every)
        self.eval_checker = RobustPeriodicChecker(10000)  # Progress every 10k episodes

        print(f"Save frequencies for 2M episode simulation:")
        print(
            f"  - Model checkpoints: every {self.config.checkpoint_every:,} episodes (~20 saves)"
        )
        print(
            f"  - Visualizations: every {self.config.visualize_every:,} episodes (~400 images)"
        )
        print(f"  - Progress updates: every 10,000 episodes (~200 updates)")

        # Create directories for organized saving
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("visualizations", exist_ok=True)

        # Initialize periodic task trackers
        self.periodic_tasks = PeriodicTaskTracker()

        # Register periodic tasks
        self.periodic_tasks.register_task(
            "target_update",
            self.config.target_update_freq,
            lambda step: self.agent.update_target_network(),
        )

        # Use individual checkers for more control
        self.target_update_checker = RobustPeriodicChecker(
            self.config.target_update_freq
        )
        self.progress_checker = RobustPeriodicChecker(200)

        # Experience collection thread
        self.collection_thread = threading.Thread(
            target=self._collect_experiences, daemon=True
        )
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
                self.coverage_improvements.append(
                    result_data["total_coverage_improvement"]
                )
                self.avg_coverage_improvements.append(
                    result_data["avg_coverage_improvement"]
                )

                # Enhanced metrics
                self.efficiency_scores.append(result_data.get("efficiency_score", 0))
                self.touching_scores.append(result_data.get("touching_score", 0))

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
                    "maps_generated": result_data.get("maps_generated", 0),
                }

            except:
                continue

    def _update_epsilon(self):
        """Update shared epsilon value with sophisticated decay for massive simulation."""
        current_episodes = len(self.episode_rewards)

        # Three-phase epsilon decay for better exploration:
        # Phase 1 (0-500k): Slow decay from 1.0 to 0.5 (heavy exploration)
        # Phase 2 (500k-1.5M): Faster decay from 0.5 to 0.05 (balanced)
        # Phase 3 (1.5M-2M): Very slow decay from 0.05 to 0.01 (exploitation)

        if current_episodes < 500000:
            # Phase 1: Slow linear decay for thorough exploration
            progress = current_episodes / 500000
            new_epsilon = 1.0 - (0.5 * progress)
        elif current_episodes < 1500000:
            # Phase 2: Exponential decay for balanced exploration/exploitation
            progress = (current_episodes - 500000) / 1000000
            # Exponential decay from 0.5 to 0.05
            new_epsilon = 0.5 * np.exp(-3.2 * progress)  # 0.5 * e^(-3.2) ≈ 0.05
        else:
            # Phase 3: Very slow decay for fine-tuning
            progress = (current_episodes - 1500000) / 500000
            new_epsilon = 0.05 - (0.04 * progress)

        new_epsilon = max(new_epsilon, self.config.epsilon_end)

        with self.epsilon_value.get_lock():
            self.epsilon_value.value = new_epsilon
            self.agent.epsilon = new_epsilon

        # Debug print every 10k episodes
        if current_episodes % 10000 == 0 and current_episodes > 0:
            phase = (
                "Phase 1"
                if current_episodes < 500000
                else "Phase 2"
                if current_episodes < 1500000
                else "Phase 3"
            )
            print(
                f"\n📊 Epsilon Update - Episode {current_episodes:,}: ε = {new_epsilon:.4f} ({phase})"
            )

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
        """Enhanced training step with Double DQN."""
        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return None

        try:
            state_batch, actions, rewards, next_states, dones = (
                self._prepare_batch_tensors_optimized(batch)
            )

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

            # Enhanced training with proper Double DQN
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    # Get current Q values
                    current_q_values = self.agent.q_network(state_batch)
                    if current_q_values.dim() == 3:
                        current_q_values = current_q_values.view(
                            current_q_values.size(0), -1
                        )
                    current_q_values = current_q_values.gather(
                        1, action_indices.unsqueeze(1)
                    ).squeeze(1)

                    # Calculate target Q values with Double DQN
                    with torch.no_grad():
                        target_q_values = torch.FloatTensor(rewards).to(self.device)

                        if next_states:
                            # Prepare next state batch
                            next_states_full = []
                            non_final_mask = []
                            for i, (s, d) in enumerate(
                                zip([e[3] for e in batch], dones)
                            ):
                                if s is not None and not d:
                                    next_states_full.append(s)
                                    non_final_mask.append(i)

                            if next_states_full:
                                next_state_batch = (
                                    self._prepare_batch_tensors_optimized(
                                        [
                                            (s, None, None, None, None)
                                            for s in next_states_full
                                        ]
                                    )[0]
                                )

                                # Double DQN: use online network to select actions
                                next_q_values = self.agent.q_network(next_state_batch)
                                if next_q_values.dim() == 3:
                                    next_q_values = next_q_values.view(
                                        next_q_values.size(0), -1
                                    )
                                next_actions = next_q_values.max(1)[1]

                                # Use target network to evaluate actions
                                next_q_values_target = self.agent.target_network(
                                    next_state_batch
                                )
                                if next_q_values_target.dim() == 3:
                                    next_q_values_target = next_q_values_target.view(
                                        next_q_values_target.size(0), -1
                                    )
                                next_q_selected = next_q_values_target.gather(
                                    1, next_actions.unsqueeze(1)
                                ).squeeze(1)

                                # Add future rewards for non-terminal states
                                for i, idx in enumerate(non_final_mask):
                                    target_q_values[idx] += (
                                        self.agent.gamma * next_q_selected[i]
                                    )

                    loss = nn.SmoothL1Loss()(current_q_values, target_q_values)

                self.agent.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.agent.optimizer)
                self.scaler.update()
            else:
                # Same logic without mixed precision
                current_q_values = self.agent.q_network(state_batch)
                if current_q_values.dim() == 3:
                    current_q_values = current_q_values.view(
                        current_q_values.size(0), -1
                    )
                current_q_values = current_q_values.gather(
                    1, action_indices.unsqueeze(1)
                ).squeeze(1)

                with torch.no_grad():
                    target_q_values = torch.FloatTensor(rewards).to(self.device)

                    # Prepare next states properly
                    next_states_full = []
                    non_final_mask = []
                    for i, (s, d) in enumerate(zip([e[3] for e in batch], dones)):
                        if s is not None and not d:
                            next_states_full.append(s)
                            non_final_mask.append(i)

                    if next_states_full:
                        next_state_batch = self._prepare_batch_tensors_optimized(
                            [(s, None, None, None, None) for s in next_states_full]
                        )[0]

                        next_q_values = self.agent.q_network(next_state_batch)
                        if next_q_values.dim() == 3:
                            next_q_values = next_q_values.view(
                                next_q_values.size(0), -1
                            )
                        next_actions = next_q_values.max(1)[1]

                        next_q_values_target = self.agent.target_network(
                            next_state_batch
                        )
                        if next_q_values_target.dim() == 3:
                            next_q_values_target = next_q_values_target.view(
                                next_q_values_target.size(0), -1
                            )
                        next_q_selected = next_q_values_target.gather(
                            1, next_actions.unsqueeze(1)
                        ).squeeze(1)

                        for i, idx in enumerate(non_final_mask):
                            target_q_values[idx] += (
                                self.agent.gamma * next_q_selected[i]
                            )

                loss = nn.SmoothL1Loss()(current_q_values, target_q_values)

                self.agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 1.0)
                self.agent.optimizer.step()

            self.training_step += 1

            # Soft update target network
            self.agent.soft_update_target_network()

            # Hard update target network periodically
            if self.target_update_checker.should_execute(self.training_step):
                self.agent.update_target_network()

            return loss.item()

        except Exception as e:
            print(f"Training step error: {e}")
            return None

    def _cleanup_memory(self):
        """Cleanup memory periodically."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train(self):
        """Main training loop with enhancements."""
        print("=" * 100)
        print("ENHANCED RANDOMIZED RADII PARALLEL TRAINING - MASSIVE SIMULATION")
        print("=" * 100)
        print(f"Episodes: {self.config.n_episodes:,}")
        print(f"Workers: {self.config.n_workers}")
        print(f"Map size: {self.config.map_size}")
        print(f"Device: {self.device}")
        print(f"\nEnhancements:")
        print(f"  - Better reward shaping with efficiency and tight packing bonuses")
        print(f"  - Enhanced heuristic agent with smart exploration")
        print(f"  - Double DQN with soft updates")
        print(
            f"  - Larger batch size ({self.config.batch_size}) and buffer ({self.config.buffer_size:,})"
        )
        print(f"  - Three-phase epsilon decay for optimal exploration:")
        print(f"    • Phase 1 (0-500k): Heavy exploration (ε: 1.0→0.5)")
        print(f"    • Phase 2 (500k-1.5M): Balanced (ε: 0.5→0.05)")
        print(f"    • Phase 3 (1.5M-2M): Fine-tuning (ε: 0.05→0.01)")
        print(f"\nSave Configuration:")
        print(f"  - Model checkpoints: every {self.config.checkpoint_every:,} episodes")
        print(f"  - Visualizations: every {self.config.visualize_every:,} episodes")
        print(f"  - Progress updates: every 10,000 episodes")
        print(f"  - All saves in: checkpoints/ and visualizations/ directories")
        print("=" * 100)

        pbar = tqdm(total=self.config.n_episodes, desc="Enhanced Training")
        last_episode_count = 0
        start_time = time.time()

        print("\n📌 Note: Epsilon decay is based on EPISODES, not training steps!")
        print(f"   Phase 1: Episodes 0-500k (ε: 1.0→0.5)")
        print(f"   Phase 2: Episodes 500k-1.5M (ε: 0.5→0.05)")
        print(f"   Phase 3: Episodes 1.5M-2M (ε: 0.05→0.01)\n")

        while len(self.episode_rewards) < self.config.n_episodes:
            current_episodes = len(self.episode_rewards)
            pbar.update(current_episodes - last_episode_count)
            last_episode_count = current_episodes

            # Update epsilon
            self._update_epsilon()

            # Train if we have enough experiences
            if (
                len(self.replay_buffer)
                > self.config.batch_size * self.config.gradient_accumulation_steps
            ):
                total_loss = 0
                num_batches = 0

                for _ in range(self.config.gradient_accumulation_steps):
                    loss = self._train_step()
                    if loss is not None:
                        total_loss += loss
                        num_batches += 1

                if num_batches > 0:
                    self.losses.append(total_loss / num_batches)

            # Update progress bar with enhanced metrics
            if current_episodes > 0:
                recent_coverage = (
                    np.mean(self.episode_coverage[-1000:])
                    if len(self.episode_coverage) >= 1000
                    else np.mean(self.episode_coverage)
                    if self.episode_coverage
                    else 0
                )
                recent_reward = (
                    np.mean(self.episode_rewards[-1000:])
                    if len(self.episode_rewards) >= 1000
                    else np.mean(self.episode_rewards)
                    if self.episode_rewards
                    else 0
                )
                recent_efficiency = (
                    np.mean(self.efficiency_scores[-1000:])
                    if len(self.efficiency_scores) >= 1000
                    else np.mean(self.efficiency_scores)
                    if self.efficiency_scores
                    else 0
                )

                with self.epsilon_value.get_lock():
                    current_epsilon = self.epsilon_value.value

                # Determine epsilon phase
                if current_episodes < 500000:
                    phase = "P1:Explore"
                elif current_episodes < 1500000:
                    phase = "P2:Balance"
                else:
                    phase = "P3:Exploit"

                pbar.set_postfix(
                    {
                        "Phase": phase,
                        "Coverage": f"{recent_coverage:.1%}",
                        "Reward": f"{recent_reward:.2f}",
                        "Efficiency": f"{recent_efficiency:.3f}",
                        "Buffer": f"{len(self.replay_buffer):,}",
                        "Loss": f"{np.mean(self.losses[-100:]):.4f}"
                        if self.losses
                        else "N/A",
                        "ε": f"{current_epsilon:.3f}",
                    }
                )

                # Debug epsilon every 1000 episodes
                if current_episodes % 1000 == 0 and current_episodes > 0:
                    print(
                        f"\nDEBUG: Episode {current_episodes}, Epsilon from shared value: {current_epsilon:.6f}, Agent epsilon: {self.agent.epsilon:.6f}"
                    )

            # Use robust periodic checkers for all periodic tasks
            if current_episodes > 0:
                # Progress updates
                if self.progress_checker.should_execute(current_episodes):
                    self._print_progress_update(current_episodes)

                # Model checkpoint saving
                if self.model_save_checker.should_execute(current_episodes):
                    self._save_checkpoint(current_episodes)
                    self._cleanup_memory()

                # Visualization saving
                if self.image_save_checker.should_execute(current_episodes):
                    self._quick_visualize(current_episodes)
                    self._save_training_progress_plot(current_episodes)

            time.sleep(0.01)

        pbar.close()

        # Final statistics
        end_time = time.time()
        training_time = end_time - start_time

        print("\n" + "=" * 100)
        print("ENHANCED RANDOMIZED RADII TRAINING COMPLETE!")
        print("=" * 100)
        print(
            f"Training time: {training_time:.2f} seconds ({training_time / 60:.1f} minutes)"
        )
        print(f"Episodes per second: {self.config.n_episodes / training_time:.2f}")
        print(f"Final average coverage: {np.mean(self.episode_coverage[-1000:]):.1%}")
        print(f"Best coverage achieved: {max(self.episode_coverage):.1%}")
        print(f"Average efficiency score: {np.mean(self.efficiency_scores):.3f}")
        print(f"Total training steps: {self.training_step:,}")

        self._save_final_model()
        self._cleanup()

    def _print_progress_update(self, episode: int):
        """Print detailed progress update."""
        if len(self.episode_rewards) < 50:
            return

        recent_coverage = np.mean(self.episode_coverage[-200:])
        recent_rewards = np.mean(self.episode_rewards[-200:])
        recent_efficiency = np.mean(self.efficiency_scores[-200:])
        best_coverage = max(self.episode_coverage)

        print(f"\n🎯 Episode {episode:,} Progress (Enhanced):")
        print(f"   Coverage: {recent_coverage:.1%} (Best: {best_coverage:.1%})")
        print(f"   Reward: {recent_rewards:.2f}")
        print(f"   Efficiency: {recent_efficiency:.3f}")
        print(f"   Training Steps: {self.training_step:,}")
        print(f"   Buffer: {len(self.replay_buffer):,}/{self.config.buffer_size:,}")

    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        try:
            checkpoint_path = f"checkpoints/enhanced_checkpoint_ep{episode:06d}.pth"

            save_data = {
                "episode": episode,
                "model_state_dict": self.agent.q_network.state_dict(),
                "optimizer_state_dict": self.agent.optimizer.state_dict(),
                "training_step": self.training_step,
                "episode_rewards": self.episode_rewards[-1000:],
                "episode_coverage": self.episode_coverage[-1000:],
                "efficiency_scores": self.efficiency_scores[-1000:],
                "config": self.config,
            }

            torch.save(save_data, checkpoint_path)
            print(f"   ✅ Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            print(f"   ❌ Error saving checkpoint: {e}")

    def _save_final_model(self):
        """Save the final trained model."""
        try:
            model_path = "enhanced_randomized_radii_final_model.pth"

            save_data = {
                "model_state_dict": self.agent.q_network.state_dict(),
                "optimizer_state_dict": self.agent.optimizer.state_dict(),
                "training_step": self.training_step,
                "episode_rewards": self.episode_rewards,
                "episode_coverage": self.episode_coverage,
                "efficiency_scores": self.efficiency_scores,
                "config": self.config,
            }

            torch.save(save_data, model_path)
            print(f"   ✅ Final model saved: {model_path}")

        except Exception as e:
            print(f"   ❌ Error saving final model: {e}")

    def _quick_visualize(self, episode: int):
        """Quick visualization."""
        try:
            save_path = f"visualizations/enhanced_strategy_ep{episode:06d}.png"

            # Create a test environment for visualization
            test_env = EnhancedRandomizedRadiiEnvironment(self.config.map_size)
            weighted_matrix = random_seeder(self.config.map_size, time_steps=100000)
            state = test_env.reset(weighted_matrix)

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Original heatmap
            axes[0, 0].imshow(test_env.original_map, cmap="hot")
            axes[0, 0].set_title("Original Heatmap")
            axes[0, 0].axis("off")

            # Track metrics
            coverage_history = [0.0]
            efficiency_history = []
            radii_used = []

            # Show placements step by step
            steps_to_show = [
                2,
                min(5, len(test_env.radii) - 1),
                min(8, len(test_env.radii) - 1),
            ]
            step_idx = 0

            for i in range(len(test_env.radii)):
                current_radius = test_env.radii[i]
                radii_used.append(current_radius)

                # Get valid actions mask
                valid_mask = np.ones(
                    (self.config.map_size, self.config.map_size), dtype=bool
                )
                valid_mask[:current_radius, :] = False
                valid_mask[-current_radius:, :] = False
                valid_mask[:, :current_radius] = False
                valid_mask[:, -current_radius:] = False

                # Get agent's action using enhanced heuristic
                heuristic_agent = EnhancedHeuristicAgent(self.config.map_size)
                state_with_info = state.copy()
                state_with_info["current_radius"] = current_radius
                state_with_info["placed_circles"] = test_env.placed_circles.copy()
                action = heuristic_agent.act(
                    state_with_info, valid_mask.flatten(), epsilon=0.0
                )  # No randomness

                # Take step
                state, reward, done, info = test_env.step(action)
                coverage_history.append(info["coverage"])
                if "efficiency" in info:
                    efficiency_history.append(info["efficiency"])

                # Visualize at specific steps
                if i + 1 in steps_to_show and step_idx < 3:
                    positions = [(0, 1), (0, 2), (1, 1)]
                    row, col = positions[step_idx]

                    axes[row, col].imshow(test_env.original_map, cmap="hot", alpha=0.6)

                    # Draw circles with different colors for different sizes
                    for j, (x, y, r) in enumerate(test_env.placed_circles):
                        # Color by size: red=large, blue=medium, green=small
                        if r >= 15:
                            color = "red"
                        elif r >= 8:
                            color = "blue"
                        else:
                            color = "green"
                        circle = plt.Circle(
                            (y, x), r, fill=False, color=color, linewidth=2
                        )
                        axes[row, col].add_patch(circle)

                    coverage_improvement = (
                        coverage_history[-1] - coverage_history[-2]
                        if len(coverage_history) > 1
                        else 0
                    )
                    axes[row, col].set_title(
                        f"Step {i + 1}: r={current_radius}, Cov {info['coverage']:.1%} (+{coverage_improvement:.1%})"
                    )
                    axes[row, col].axis("off")

                    step_idx += 1

                if done:
                    break

            # Final result
            axes[1, 0].imshow(test_env.original_map, cmap="hot", alpha=0.6)
            for j, (x, y, r) in enumerate(test_env.placed_circles):
                if r >= 15:
                    color = "red"
                elif r >= 8:
                    color = "blue"
                else:
                    color = "green"
                circle = plt.Circle((y, x), r, fill=False, color=color, linewidth=2)
                axes[1, 0].add_patch(circle)

            axes[1, 0].set_title(f"Final: {info['coverage']:.1%} coverage")
            axes[1, 0].axis("off")

            # Coverage progress
            axes[1, 1].plot(coverage_history, "g-", linewidth=2, marker="o")
            axes[1, 1].set_title("Coverage Progress")
            axes[1, 1].set_xlabel("Circle Placement")
            axes[1, 1].set_ylabel("Coverage")
            axes[1, 1].grid(True)

            # Enhanced info with efficiency
            config_text = f"Episode: {episode}\n"
            config_text += f"Circles: {len(test_env.radii)}\n"
            config_text += f"Radii: {test_env.radii}\n"
            config_text += f"Achieved: {info['coverage']:.1%}\n"
            config_text += (
                f"Avg Efficiency: {np.mean(efficiency_history):.3f}\n"
                if efficiency_history
                else ""
            )
            config_text += f"Training Steps: {self.training_step:,}\n"
            config_text += f"Buffer: {len(self.replay_buffer):,}\n"

            # Add recent performance stats
            if len(self.episode_coverage) > 100:
                recent_coverage = np.mean(self.episode_coverage[-100:])
                recent_reward = np.mean(self.episode_rewards[-100:])
                recent_efficiency = (
                    np.mean(self.efficiency_scores[-100:])
                    if len(self.efficiency_scores) > 100
                    else 0
                )
                config_text += f"Recent Avg Coverage: {recent_coverage:.1%}\n"
                config_text += f"Recent Avg Reward: {recent_reward:.1f}\n"
                config_text += f"Recent Avg Efficiency: {recent_efficiency:.3f}"

            axes[1, 2].text(
                0.05,
                0.95,
                config_text,
                fontsize=10,
                transform=axes[1, 2].transAxes,
                verticalalignment="top",
                fontfamily="monospace",
            )
            axes[1, 2].set_title("Configuration & Stats")
            axes[1, 2].axis("off")

            # Add legend for circle colors
            legend_text = (
                "Circle Sizes:\n🔴 Large (≥15)\n🔵 Medium (8-14)\n🟢 Small (<8)"
            )
            axes[0, 1].text(
                0.02,
                0.02,
                legend_text,
                fontsize=10,
                transform=axes[0, 1].transAxes,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            plt.suptitle(
                f"Enhanced Randomized Radii Agent - Episode {episode} ({len(test_env.radii)} circles)"
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"📸 Visualization saved: {save_path}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Visualization error: {e}")
            import traceback

            traceback.print_exc()

    def _save_training_progress_plot(self, episode: int):
        """Save training progress plot with enhanced metrics."""
        try:
            save_path = f"visualizations/enhanced_progress_ep{episode:06d}.png"

            episodes = np.arange(len(self.episode_coverage))
            plt.figure(figsize=(15, 10))

            # Coverage
            plt.subplot(2, 2, 1)
            plt.plot(episodes, self.episode_coverage, "b-", linewidth=2)
            plt.axhline(
                y=0.37, color="r", linestyle="--", alpha=0.5, label="Previous plateau"
            )
            plt.title("Coverage Over Episodes")
            plt.xlabel("Episode")
            plt.ylabel("Coverage (%)")
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 1.05)

            # Rewards
            plt.subplot(2, 2, 2)
            plt.plot(episodes, self.episode_rewards, "g-", linewidth=2)
            plt.title("Reward Over Episodes")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)

            # Efficiency scores
            plt.subplot(2, 2, 3)
            plt.plot(episodes, self.efficiency_scores, "m-", linewidth=2)
            plt.title("Efficiency Score Over Episodes")
            plt.xlabel("Episode")
            plt.ylabel("Efficiency")
            plt.grid(True)

            # Loss
            if self.losses:
                plt.subplot(2, 2, 4)
                plt.plot(self.losses, "r-", linewidth=1, alpha=0.7)
                plt.title("Training Loss")
                plt.xlabel("Training Step")
                plt.ylabel("Loss")
                plt.yscale("log")
                plt.grid(True)

            plt.suptitle(f"Enhanced Training Progress - Episode {episode}")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"📊 Progress plot saved: {save_path}")
        except Exception as e:
            print(f"❌ Error saving progress plot: {e}")

    def _cleanup(self):
        """Clean up resources."""
        for worker in self.workers:
            worker.terminate()

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Random DQN Training")
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000000,
        help="Number of episodes to train (default: 2,000,000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers (default: auto-detect)",
    )
    args = parser.parse_args()

    # Detect system capabilities
    n_cores = mp.cpu_count()
    n_workers = (
        args.workers if args.workers else min(n_cores - 1, 64)
    )  # Use up to 64 cores

    print(f"System: {n_cores} cores")
    print(f"Using {n_workers} workers for enhanced training")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Configuration
    config = EnhancedRandomizedRadiiConfig(
        n_episodes=args.episodes,
        n_workers=n_workers,
        map_size=128,
    )

    # Create trainer and start training
    trainer = EnhancedRandomizedRadiiTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
