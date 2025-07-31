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
    """Environment with randomized number of circles and their sizes."""
    
    def __init__(self, map_size=128):
        self.previous_coverage = 0.0
        self.map_size = map_size
        # Don't call super().__init__() yet - we'll set radii first
        
        # Initialize other attributes that parent __init__ expects
        self.placed_circles = []
        self.placement_order = []
        self.coverage_history = []
        self.current_radius_idx = 0
        self.total_weight_collected = 0
        
        # Generate initial random radii
        self._generate_random_radii()
        
        # Now call parent init with our custom radii
        super(AdvancedCirclePlacementEnv, self).__init__()  # Skip AdvancedCirclePlacementEnv.__init__
        
        # Initialize the components manually
        from src.algorithms.dqn_agent import HeatmapFeatureExtractor
        self.feature_extractor = HeatmapFeatureExtractor(map_size)
        
    def _generate_random_radii(self):
        """Generate random number of circles with random sizes - OPTIMIZED."""
        # Random number of circles (3-15 circles)
        n_circles = np.random.randint(3, 16)
        
        # Generate random radii with some constraints - VECTORIZED
        min_radius = 2
        max_radius = min(20, self.map_size // 8)
        
        # Vectorized generation for speed
        radii = []
        small_medium_count = int(n_circles * 0.7)  # 70% small-medium
        large_count = n_circles - small_medium_count  # 30% large
        
        # Generate small-medium circles
        if small_medium_count > 0:
            small_radii = np.random.randint(min_radius, max_radius // 2 + 1, size=small_medium_count)
            radii.extend(small_radii.tolist())
        
        # Generate large circles
        if large_count > 0:
            large_radii = np.random.randint(max_radius // 2, max_radius + 1, size=large_count)
            radii.extend(large_radii.tolist())
        
        # Sort radii in descending order (place large circles first)
        self.radii = sorted(radii, reverse=True)
        self.n_circles = len(self.radii)
        
        # Calculate theoretical maximum coverage for normalization - CACHED
        self._calculate_max_theoretical_coverage_fast()
        
    def _calculate_max_theoretical_coverage(self):
        """Calculate theoretical maximum coverage for normalization."""
        # Simple approximation: sum of circle areas (with some overlap discount)
        total_area = sum(np.pi * r * r for r in self.radii)
        map_area = self.map_size * self.map_size
        
        # Apply overlap discount (circles can't achieve 100% efficiency)
        overlap_factor = 0.8  # Assume 20% area loss due to overlaps and boundaries
        self.max_theoretical_coverage = min(1.0, (total_area * overlap_factor) / map_area)
    
    def _calculate_max_theoretical_coverage_fast(self):
        """OPTIMIZED: Calculate theoretical maximum coverage using vectorized operations."""
        # Vectorized calculation for speed
        radii_array = np.array(self.radii, dtype=np.float32)
        total_area = np.sum(np.pi * radii_array * radii_array)
        map_area = self.map_size * self.map_size
        
        # Apply overlap discount (circles can't achieve 100% efficiency)
        overlap_factor = 0.8
        self.max_theoretical_coverage = min(1.0, (total_area * overlap_factor) / map_area)
        
    def reset(self, weighted_matrix=None):
        """Reset environment with new random radii configuration."""
        if weighted_matrix is None:
            weighted_matrix = random_seeder(self.map_size, time_steps=100000)
        
        # Generate new random radii for each episode
        self._generate_random_radii()
        
        # Reset all state
        self.placed_circles = []
        self.placement_order = []
        self.coverage_history = []
        self.current_radius_idx = 0
        self.total_weight_collected = 0
        self.previous_coverage = 0.0
        
        # Clear caches for optimization
        if hasattr(self, '_cached_total_area'):
            delattr(self, '_cached_total_area')
        if hasattr(self, '_cached_areas'):
            delattr(self, '_cached_areas')
        
        # Set up maps
        self.original_map = weighted_matrix.copy()
        self.current_map = weighted_matrix.copy()
        
        return self._get_enhanced_state()
    
    def _get_enhanced_state(self):
        """Get enhanced state with normalization for variable radii."""
        # Create placed mask
        placed_mask = np.zeros_like(self.current_map)
        for x, y, r in self.placed_circles:
            for i in range(max(0, int(x - r)), min(self.map_size, int(x + r + 1))):
                for j in range(max(0, int(y - r)), min(self.map_size, int(y + r + 1))):
                    if (i - x) ** 2 + (j - y) ** 2 <= r**2:
                        placed_mask[i, j] = 1
        
        # Calculate value density
        if self.original_map.sum() > 0:
            value_density = self.current_map / self.original_map.sum()
        else:
            value_density = np.zeros_like(self.current_map)
        
        # Extract features for current radius
        current_radius = self.radii[self.current_radius_idx] if self.current_radius_idx < len(self.radii) else self.radii[-1]
        raw_features = self.feature_extractor.extract_features(self.current_map, current_radius)
        
        # Normalize features based on current configuration
        features = self._normalize_features(raw_features, current_radius)
        
        return {
            "current_map": self.current_map,
            "placed_mask": placed_mask,
            "value_density": value_density,
            "features": features,
            "raw_features": raw_features,
            # Add configuration info for the agent
            "current_radius": current_radius,
            "remaining_circles": len(self.radii) - self.current_radius_idx,
            "total_circles": len(self.radii),
            "progress": self.current_radius_idx / len(self.radii),
            "max_theoretical_coverage": self.max_theoretical_coverage,
        }
    
    def _normalize_features(self, raw_features, current_radius):
        """OPTIMIZED: Normalize features for variable configurations."""
        features = np.zeros(10, dtype=np.float32)  # Fixed size feature vector
        
        # Cache frequently used values
        original_sum = self.original_map.sum()
        current_sum = self.current_map.sum()
        original_max = self.original_map.max()
        
        # Basic coverage and density features
        current_coverage = 1 - (current_sum / original_sum) if original_sum > 0 else 0
        features[0] = current_coverage
        features[1] = current_coverage / max(self.max_theoretical_coverage, 0.01)  # Normalized coverage
        
        # Circle configuration features - OPTIMIZED
        features[2] = current_radius * 0.05  # current_radius / 20.0
        features[3] = len(self.radii) / 15.0  # Normalized number of circles
        features[4] = self.current_radius_idx / max(len(self.radii) - 1, 1)  # Progress through circles
        
        # Remaining potential - VECTORIZED
        if not hasattr(self, '_cached_total_area'):
            radii_array = np.array(self.radii, dtype=np.float32)
            self._cached_areas = np.pi * radii_array * radii_array
            self._cached_total_area = np.sum(self._cached_areas)
        
        remaining_area = np.sum(self._cached_areas[self.current_radius_idx:])
        features[5] = remaining_area / max(self._cached_total_area, 1)  # Remaining potential
        
        # Map characteristics - OPTIMIZED
        if current_sum > 0:
            features[6] = self.current_map.mean() / original_max
            features[7] = self.current_map.std() / original_max
            features[8] = self.current_map.max() / original_max
        
        # Cluster information (if available)
        if "num_clusters" in raw_features:
            features[9] = min(raw_features["num_clusters"] * 0.1, 1.0)  # /10.0 optimized
        
        return features
    
    def step(self, action):
        """Step with IMPROVED reward function for better learning."""
        x, y = action
        radius = self.radii[self.current_radius_idx]
        
        # Store current coverage before action
        current_coverage = 1 - (self.current_map.sum() / self.original_map.sum())
        
        # Calculate base collection value
        included_weight = compute_included(self.current_map, x, y, radius)
        
        # OVERLAP PENALTY - Check for overlaps with existing circles
        overlap_penalty = 0.0
        for px, py, pr in self.placed_circles:
            distance = np.sqrt((x - px)**2 + (y - py)**2)
            min_distance = radius + pr
            if distance < min_distance:
                # Overlap detected - heavy penalty
                overlap_ratio = (min_distance - distance) / min_distance
                overlap_penalty += overlap_ratio * 2.0  # Strong penalty
        
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
        coverage_improvement = new_coverage - current_coverage
        
        # IMPROVED REWARD FUNCTION
        reward = 0.0
        
        if included_weight > 0:
            # 1. VALUE COLLECTION REWARD - Strong focus on collecting high values
            circle_area = np.pi * radius * radius
            max_possible_in_circle = circle_area * self.original_map.max()
            value_efficiency = included_weight / max(max_possible_in_circle, 1)
            
            # Scale by circle importance (larger circles should be more selective)
            circle_importance = radius / max(self.radii)
            value_reward = value_efficiency * (5.0 + 5.0 * circle_importance)
            reward += value_reward
            
            # 2. COVERAGE IMPROVEMENT REWARD - Normalized by theoretical max
            coverage_reward = (coverage_improvement / max(self.max_theoretical_coverage, 0.01)) * 8.0
            reward += coverage_reward
            
            # 3. HIGH-VALUE TARGETING BONUS - Reward for targeting hot spots
            # Calculate average value in the circle area
            avg_value_in_circle = included_weight / max(circle_area, 1)
            map_avg_value = self.original_map.mean()
            if map_avg_value > 0:
                hotspot_bonus = (avg_value_in_circle / map_avg_value - 1.0) * 2.0
                reward += max(0, hotspot_bonus)  # Only positive bonus
            
            # 4. STRATEGIC PLACEMENT BONUS - Early circles should be more selective
            progress = self.current_radius_idx / len(self.radii)
            if progress < 0.3:  # First 30% of circles (usually the largest)
                # Reward high-value placement more for early circles
                selectivity_bonus = value_efficiency * 3.0
                reward += selectivity_bonus
            
            # 5. REMAINING POTENTIAL BONUS - Consider future circles
            remaining_circles = len(self.radii) - self.current_radius_idx - 1
            if remaining_circles > 0:
                # Bonus for leaving good opportunities for future circles
                remaining_value = self.current_map.sum()
                total_remaining_area = sum(np.pi * r * r for r in self.radii[self.current_radius_idx + 1:])
                if total_remaining_area > 0:
                    future_potential = remaining_value / total_remaining_area
                    potential_bonus = min(future_potential / self.original_map.max(), 1.0) * 1.0
                    reward += potential_bonus
        
        else:
            # HEAVY PENALTY for placing circles in empty areas
            circle_importance = radius / sum(self.radii)
            wasted_penalty = 3.0 * (1.0 + 2.0 * circle_importance)  # Stronger penalty
            reward -= wasted_penalty
        
        # 6. OVERLAP PENALTY - Apply the calculated overlap penalty
        reward -= overlap_penalty
        
        # 7. BOUNDARY PENALTY - Penalize circles too close to edges
        boundary_penalty = 0.0
        edge_distance = min(x, y, self.map_size - x, self.map_size - y)
        if edge_distance < radius * 1.2:  # Too close to edge
            boundary_ratio = (radius * 1.2 - edge_distance) / (radius * 1.2)
            boundary_penalty = boundary_ratio * 1.0
        reward -= boundary_penalty
        
        # Move to next radius
        self.current_radius_idx += 1
        done = self.current_radius_idx >= len(self.radii)
        
        # Update coverage tracking
        self.coverage_history.append(new_coverage)
        self.previous_coverage = new_coverage
        
        info = {
            "coverage": new_coverage,
            "coverage_improvement": coverage_improvement,
            "normalized_coverage": new_coverage / max(self.max_theoretical_coverage, 0.01),
            "included_weight": included_weight,
            "efficiency": value_efficiency if included_weight > 0 else 0,
            "overlap_penalty": overlap_penalty,
            "boundary_penalty": boundary_penalty,
            "n_circles": len(self.radii),
            "max_theoretical_coverage": self.max_theoretical_coverage,
            "radii_config": self.radii.copy(),
        }
        
        if done:
            # FINAL PERFORMANCE EVALUATION
            normalized_final_coverage = new_coverage / max(self.max_theoretical_coverage, 0.01)
            
            # Strong final bonuses for good performance
            if normalized_final_coverage > 0.90:
                reward += 15.0  # Excellent performance
            elif normalized_final_coverage > 0.80:
                reward += 10.0  # Very good performance
            elif normalized_final_coverage > 0.70:
                reward += 5.0   # Good performance
            elif normalized_final_coverage > 0.60:
                reward += 2.0   # Acceptable performance
            elif normalized_final_coverage > 0.50:
                reward += 1.0   # Marginal performance
            else:
                # Poor performance penalty
                reward -= 5.0 * (0.50 - normalized_final_coverage)
            
            # Bonus for efficient use of all circles (no major overlaps)
            if overlap_penalty < 1.0:  # Low total overlap
                reward += 3.0
            
            return None, reward, done, info
        
        return self._get_enhanced_state(), reward, done, info
    
    def get_valid_actions_mask(self):
        """Get valid actions mask for current radius."""
        radius = self.radii[self.current_radius_idx]
        mask = np.ones((self.map_size, self.map_size), dtype=bool)
        
        # Basic boundary constraints
        mask[:int(radius), :] = False
        mask[-int(radius):, :] = False
        mask[:, :int(radius)] = False
        mask[:, -int(radius):] = False
        
        # Avoid placing circles where there's no value
        value_threshold = self.original_map.max() * 0.01  # 1% of max value
        low_value_mask = self.current_map < value_threshold
        mask[low_value_mask] = False
        
        return mask.astype(float)


@dataclass
class RandomizedRadiiConfig:
    """Configuration for randomized radii training."""
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


class RandomizedHeuristicAgent:
    """Heuristic agent that adapts to different radii configurations."""
    
    def __init__(self, map_size: int):
        self.map_size = map_size
        
    def act(self, state_dict, env, valid_mask=None, epsilon=0.5):
        """IMPROVED: Choose action with better targeting and overlap avoidance."""
        current_radius = state_dict.get("current_radius", 5)
        remaining_circles = state_dict.get("remaining_circles", 1)
        progress = state_dict.get("progress", 0.5)
        current_map = state_dict["current_map"]
        
        if np.random.random() < epsilon:
            # SMART EXPLORATION with overlap avoidance
            if valid_mask is not None:
                valid_positions = np.argwhere(valid_mask > 0.5)
                if len(valid_positions) > 0:
                    weights = []
                    for pos in valid_positions:
                        x, y = pos
                        
                        # Calculate potential value in circle
                        local_value = 0
                        for i in range(max(0, x - current_radius), min(self.map_size, x + current_radius + 1)):
                            for j in range(max(0, y - current_radius), min(self.map_size, y + current_radius + 1)):
                                if (i - x) ** 2 + (j - y) ** 2 <= current_radius**2:
                                    local_value += current_map[i, j]
                        
                        # Check for overlaps with existing circles
                        overlap_penalty = 0
                        for px, py, pr in env.placed_circles:
                            distance = np.sqrt((x - px)**2 + (y - py)**2)
                            min_distance = current_radius + pr
                            if distance < min_distance * 1.1:  # Avoid near-overlaps too
                                overlap_penalty += (min_distance * 1.1 - distance) / (min_distance * 1.1)
                        
                        # Weight calculation with overlap avoidance
                        base_weight = local_value
                        
                        # Configuration-based adjustment
                        if progress < 0.3:  # Early circles - be very selective
                            base_weight *= (1.0 + 1.0 * (current_radius / 20.0))
                        elif progress < 0.7:  # Mid circles - balanced
                            base_weight *= (1.0 + 0.5 * (current_radius / 20.0))
                        else:  # Late circles - opportunistic
                            base_weight *= (1.0 + 0.2 / max(remaining_circles, 1))
                        
                        # Apply overlap penalty
                        final_weight = max(0.01, base_weight - overlap_penalty * base_weight * 2.0)
                        weights.append(final_weight)
                    
                    if sum(weights) > 0:
                        weights = np.array(weights)
                        weights = weights / weights.sum()
                        # Add some temperature to the selection for exploration
                        weights = weights ** 1.5  # Make selection more peaked
                        weights = weights / weights.sum()
                        idx = np.random.choice(len(valid_positions), p=weights)
                        return tuple(valid_positions[idx])
            
            return (np.random.randint(current_radius, self.map_size - current_radius), 
                   np.random.randint(current_radius, self.map_size - current_radius))
        else:
            # GREEDY: Find best position with overlap avoidance and hotspot targeting
            best_score = -float('inf')
            best_action = None
            
            # More samples for larger circles (they need to be more careful)
            n_samples = max(20, min(50, int(20 + current_radius * 2)))
            
            # First, find hotspots in the map
            hotspot_threshold = current_map.mean() + current_map.std()
            hotspots = np.argwhere(current_map > hotspot_threshold)
            
            # Sample around hotspots (70% of samples) and randomly (30% of samples)
            hotspot_samples = int(n_samples * 0.7)
            random_samples = n_samples - hotspot_samples
            
            candidates = []
            
            # Sample around hotspots
            if len(hotspots) > 0 and hotspot_samples > 0:
                for _ in range(hotspot_samples):
                    # Pick a random hotspot
                    hotspot_idx = np.random.randint(len(hotspots))
                    hx, hy = hotspots[hotspot_idx]
                    
                    # Sample around the hotspot
                    search_radius = min(current_radius * 2, 20)
                    x = np.clip(hx + np.random.randint(-search_radius, search_radius + 1), 
                               current_radius, self.map_size - current_radius - 1)
                    y = np.clip(hy + np.random.randint(-search_radius, search_radius + 1), 
                               current_radius, self.map_size - current_radius - 1)
                    candidates.append((x, y))
            
            # Random samples
            for _ in range(random_samples):
                x = np.random.randint(current_radius, self.map_size - current_radius)
                y = np.random.randint(current_radius, self.map_size - current_radius)
                candidates.append((x, y))
            
            # Evaluate all candidates
            for x, y in candidates:
                # Calculate value collected
                local_value = 0
                for i in range(max(0, x - current_radius), min(self.map_size, x + current_radius + 1)):
                    for j in range(max(0, y - current_radius), min(self.map_size, y + current_radius + 1)):
                        if (i - x) ** 2 + (j - y) ** 2 <= current_radius**2:
                            local_value += current_map[i, j]
                
                # Calculate overlap penalty
                overlap_penalty = 0
                for px, py, pr in env.placed_circles:
                    distance = np.sqrt((x - px)**2 + (y - py)**2)
                    min_distance = current_radius + pr
                    if distance < min_distance:
                        overlap_ratio = (min_distance - distance) / min_distance
                        overlap_penalty += overlap_ratio * local_value * 3.0  # Heavy penalty
                
                # Calculate boundary penalty
                edge_distance = min(x, y, self.map_size - x, self.map_size - y)
                boundary_penalty = 0
                if edge_distance < current_radius * 1.2:
                    boundary_ratio = (current_radius * 1.2 - edge_distance) / (current_radius * 1.2)
                    boundary_penalty = boundary_ratio * local_value * 0.5
                
                # Calculate hotspot bonus
                circle_area = np.pi * current_radius * current_radius
                avg_value_in_circle = local_value / max(circle_area, 1)
                map_avg = current_map.mean()
                hotspot_bonus = 0
                if map_avg > 0:
                    hotspot_bonus = max(0, (avg_value_in_circle / map_avg - 1.0) * local_value * 0.5)
                
                # Configuration-aware scoring
                score = local_value + hotspot_bonus - overlap_penalty - boundary_penalty
                
                # Early circles should be more selective about high-value areas
                if progress < 0.3:
                    score += hotspot_bonus * 2.0  # Double hotspot bonus for early circles
                
                if score > best_score:
                    best_score = score
                    best_action = (x, y)
            
            if best_action is not None:
                return best_action
            
            # Fallback
            return (np.random.randint(current_radius, self.map_size - current_radius), 
                   np.random.randint(current_radius, self.map_size - current_radius))


def randomized_radii_worker_process(worker_id: int, config: RandomizedRadiiConfig,
                                   control_queue: mp.Queue, result_queue: mp.Queue,
                                   epsilon_value: mp.Value):
    """Worker process with randomized radii environment."""
    
    # Create randomized radii environment and agent
    env = RandomizedRadiiEnvironment(map_size=config.map_size)
    agent = RandomizedHeuristicAgent(config.map_size)
    
    # Worker statistics
    episodes_completed = 0
    maps_generated = 0
    radii_configs_generated = 0
    
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
            
            # Generate NEW random map for EACH episode (ensures generalization!)
            weighted_matrix = random_seeder(config.map_size, time_steps=100000)
            maps_generated += 1
            
            # Reset environment (this will generate new random radii configuration!)
            state = env.reset(weighted_matrix)
            radii_configs_generated += 1
            
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
            
            # Update worker statistics
            episodes_completed += 1
            
            # Send results efficiently
            result_data = {
                "worker_id": worker_id,
                "experiences": experiences,
                "episode_reward": episode_reward,
                "coverage": info["coverage"],
                "normalized_coverage": info["normalized_coverage"],
                "total_coverage_improvement": sum(coverage_improvements),
                "avg_coverage_improvement": np.mean(coverage_improvements) if coverage_improvements else 0,
                "episodes_completed": episodes_completed,
                "maps_generated": maps_generated,
                "radii_configs_generated": radii_configs_generated,
                "radii_config": info["radii_config"],
                "n_circles": info["n_circles"],
                "max_theoretical_coverage": info["max_theoretical_coverage"],
                "map_diversity_stats": {
                    "map_mean": float(weighted_matrix.mean()),
                    "map_std": float(weighted_matrix.std()),
                    "map_max": float(weighted_matrix.max()),
                    "map_min": float(weighted_matrix.min()),
                }
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


class RandomizedRadiiTrainer:
    """Trainer with randomized radii configurations."""
    
    def __init__(self, config: RandomizedRadiiConfig):
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
                target=randomized_radii_worker_process,
                args=(i, config, self.control_queues[i], self.result_queue, self.epsilon_value)
            )
            worker.start()
            self.workers.append(worker)
        
        # Training metrics - now tracking randomized configurations
        self.episode_rewards = []
        self.episode_coverage = []
        self.normalized_coverage = []
        self.coverage_improvements = []
        self.avg_coverage_improvements = []
        self.losses = []
        self.training_step = 0
        
        # Randomized configuration tracking
        self.map_diversity_stats = []
        self.radii_configs = []
        self.n_circles_history = []
        self.max_theoretical_coverage_history = []
        self.worker_stats = {}
        
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
                self.normalized_coverage.append(result_data["normalized_coverage"])
                self.coverage_improvements.append(result_data["total_coverage_improvement"])
                self.avg_coverage_improvements.append(result_data["avg_coverage_improvement"])
                
                # Track randomized configurations
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
                    "maps_generated": result_data.get("maps_generated", 0),
                    "radii_configs_generated": result_data.get("radii_configs_generated", 0)
                }
                
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
    
    def train(self):
        """Randomized radii training loop."""
        print("=" * 100)
        print(f"RANDOMIZED RADII PARALLEL TRAINING WITH {self.config.n_workers} WORKERS")
        print("=" * 100)
        print(f"Target episodes: {self.config.n_episodes:,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Buffer size: {self.config.buffer_size:,}")
        print(f"Epsilon decay: {self.config.epsilon_start} → {self.config.epsilon_end} over {self.config.epsilon_decay_episodes:,} episodes")
        print(f"Configuration: RANDOMIZED radii (3-15 circles, sizes 2-20)")
        print(f"Reward Function: Normalized Coverage-Aligned")
        print("=" * 100)
        
        pbar = tqdm(total=self.config.n_episodes, desc="Randomized Radii Training")
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
                recent_norm_coverage = np.mean(self.normalized_coverage[-1000:]) if len(self.normalized_coverage) >= 1000 else np.mean(self.normalized_coverage) if self.normalized_coverage else 0
                recent_reward = np.mean(self.episode_rewards[-1000:]) if len(self.episode_rewards) >= 1000 else np.mean(self.episode_rewards) if self.episode_rewards else 0
                recent_n_circles = np.mean(self.n_circles_history[-1000:]) if len(self.n_circles_history) >= 1000 else np.mean(self.n_circles_history) if self.n_circles_history else 0
                
                with self.epsilon_value.get_lock():
                    current_epsilon = self.epsilon_value.value
                
                pbar.set_postfix({
                    "Coverage": f"{recent_coverage:.1%}",
                    "NormCov": f"{recent_norm_coverage:.1%}",
                    "Reward": f"{recent_reward:.2f}",
                    "AvgCircles": f"{recent_n_circles:.1f}",
                    "Buffer": f"{len(self.replay_buffer):,}",
                    "Loss": f"{np.mean(self.losses[-100:]):.4f}" if self.losses else "N/A",
                    "Epsilon": f"{current_epsilon:.3f}"
                })
            
            # Periodic evaluation and visualization
            if current_episodes > 0 and current_episodes % self.config.visualize_every == 0:
                self._evaluate_and_visualize(current_episodes)
            
            # More frequent visualization (every 100 episodes)
            if current_episodes > 0 and current_episodes % 100 == 0:
                self._quick_visualize(current_episodes)
            
            # More frequent progress updates (every 200 episodes)
            if current_episodes > 0 and current_episodes % 200 == 0:
                self._print_progress_update(current_episodes)
            
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
        print("RANDOMIZED RADII TRAINING COMPLETE!")
        print("=" * 100)
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        print(f"Episodes per second: {self.config.n_episodes / training_time:.2f}")
        print(f"Final average coverage: {np.mean(self.episode_coverage[-1000:]):.1%}")
        print(f"Final average normalized coverage: {np.mean(self.normalized_coverage[-1000:]):.1%}")
        print(f"Best coverage achieved: {max(self.episode_coverage):.1%}")
        print(f"Average circles per episode: {np.mean(self.n_circles_history):.1f}")
        print(f"Total training steps: {self.training_step:,}")
        
        # Show reward-coverage correlation
        if len(self.episode_rewards) > 100:
            correlation = np.corrcoef(self.episode_rewards[-1000:], self.normalized_coverage[-1000:])[0, 1]
            print(f"Final reward-normalized coverage correlation: {correlation:.3f}")
        
        self._save_final_model()
        self._cleanup()
    
    def _print_progress_update(self, episode: int):
        """Print progress update every 200 episodes."""
        if len(self.episode_rewards) < 50:  # Need some data
            return
            
        recent_coverage = np.mean(self.episode_coverage[-200:]) if len(self.episode_coverage) >= 200 else np.mean(self.episode_coverage)
        recent_norm_coverage = np.mean(self.normalized_coverage[-200:]) if len(self.normalized_coverage) >= 200 else np.mean(self.normalized_coverage)
        recent_rewards = np.mean(self.episode_rewards[-200:]) if len(self.episode_rewards) >= 200 else np.mean(self.episode_rewards)
        recent_cov_imp = np.mean(self.avg_coverage_improvements[-200:]) if len(self.avg_coverage_improvements) >= 200 else np.mean(self.avg_coverage_improvements)
        best_coverage = max(self.episode_coverage)
        best_norm_coverage = max(self.normalized_coverage)
        
        # Configuration statistics
        recent_configs = self.radii_configs[-200:] if len(self.radii_configs) >= 200 else self.radii_configs
        recent_n_circles = self.n_circles_history[-200:] if len(self.n_circles_history) >= 200 else self.n_circles_history
        recent_max_theoretical = self.max_theoretical_coverage_history[-200:] if len(self.max_theoretical_coverage_history) >= 200 else self.max_theoretical_coverage_history
        
        # Calculate reward-coverage correlation
        correlation = 0.0
        if len(self.episode_rewards) > 100:
            correlation = np.corrcoef(self.episode_rewards[-500:], self.normalized_coverage[-500:])[0, 1]
        
        with self.epsilon_value.get_lock():
            current_epsilon = self.epsilon_value.value
        
        print(f"\n🎯 Episode {episode:,} Progress (IMPROVED Randomized Radii):")
        print(f"   Coverage: {recent_coverage:.1%} (Best: {best_coverage:.1%})")
        print(f"   Normalized Coverage: {recent_norm_coverage:.1%} (Best: {best_norm_coverage:.1%})")
        print(f"   Reward: {recent_rewards:.2f}")
        print(f"   Coverage Improvement: {recent_cov_imp:.3f}")
        print(f"   Reward-NormCoverage Correlation: {correlation:.3f}")
        print(f"   Epsilon: {current_epsilon:.3f}")
        print(f"   Training Steps: {self.training_step:,}")
        print(f"   Buffer: {len(self.replay_buffer):,}/{self.config.buffer_size:,}")
        
        # Additional debugging info
        if len(self.episode_rewards) > 10:
            recent_10_rewards = self.episode_rewards[-10:]
            recent_10_coverage = self.episode_coverage[-10:]
            print(f"   📈 Recent trend: Reward {np.mean(recent_10_rewards[-5:]):.2f} vs {np.mean(recent_10_rewards[:5]):.2f}")
            print(f"   📈 Recent trend: Coverage {np.mean(recent_10_coverage[-5:]):.1%} vs {np.mean(recent_10_coverage[:5]):.1%}")
        
        # Configuration diversity stats
        if recent_n_circles:
            print(f"   🔄 Configuration Diversity:")
            print(f"      Avg circles per episode: {np.mean(recent_n_circles):.1f} ± {np.std(recent_n_circles):.1f}")
            print(f"      Circle range: {min(recent_n_circles)} - {max(recent_n_circles)}")
            print(f"      Avg theoretical max coverage: {np.mean(recent_max_theoretical):.1%}")
        
        # Show map diversity stats
        if self.map_diversity_stats:
            recent_maps = self.map_diversity_stats[-200:] if len(self.map_diversity_stats) >= 200 else self.map_diversity_stats
            map_means = [m["map_mean"] for m in recent_maps]
            map_stds = [m["map_std"] for m in recent_maps]
            map_maxes = [m["map_max"] for m in recent_maps]
            
            print(f"   🗺️  Map Diversity (recent {len(recent_maps)} maps):")
            print(f"      Mean values: {np.mean(map_means):.2f} ± {np.std(map_means):.2f}")
            print(f"      Std values: {np.mean(map_stds):.2f} ± {np.std(map_stds):.2f}")
            print(f"      Max values: {np.mean(map_maxes):.2f} ± {np.std(map_maxes):.2f}")
        
        # Show worker stats
        if self.worker_stats:
            total_episodes = sum(stats["episodes_completed"] for stats in self.worker_stats.values())
            total_maps = sum(stats["maps_generated"] for stats in self.worker_stats.values())
            total_configs = sum(stats["radii_configs_generated"] for stats in self.worker_stats.values())
            active_workers = len(self.worker_stats)
            print(f"   👥 Workers: {active_workers} active")
            print(f"      {total_maps:,} unique maps, {total_configs:,} unique radii configs generated")
        
        # Show alignment status
        if correlation > 0.7:
            print(f"   ✅ Strong reward-coverage alignment!")
        elif correlation > 0.3:
            print(f"   ⚠️  Moderate alignment")
        else:
            print(f"   ❌ Weak alignment")
    
    def visualize_strategy(self, episode, save_path="randomized_strategy.png"):
        """Visualize the randomized radii agent's strategy."""
        # Use randomized radii environment for testing
        test_env = RandomizedRadiiEnvironment(self.config.map_size)
        weighted_matrix = random_seeder(self.config.map_size, time_steps=100000)
        state = test_env.reset(weighted_matrix)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original heatmap
        axes[0, 0].imshow(test_env.original_map, cmap="hot")
        axes[0, 0].set_title("Original Heatmap")
        axes[0, 0].axis("off")
        
        # Track coverage improvements and configuration
        coverage_history = [0.0]
        radii_used = []
        
        # Show placements step by step
        steps_to_show = [2, min(5, len(test_env.radii)-1), min(8, len(test_env.radii)-1)]
        step_idx = 0
        
        for i in range(len(test_env.radii)):
            valid_mask = test_env.get_valid_actions_mask()
            current_radius = test_env.radii[i]
            radii_used.append(current_radius)
            
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
                
                # Draw circles with different colors for different sizes
                for j, (x, y, r) in enumerate(test_env.placed_circles):
                    # Color by size: red=large, blue=medium, green=small
                    if r >= 15:
                        color = 'red'
                    elif r >= 8:
                        color = 'blue'
                    else:
                        color = 'green'
                    circle = plt.Circle((y, x), r, fill=False, color=color, linewidth=2)
                    axes[row, col].add_patch(circle)
                
                coverage_improvement = coverage_history[-1] - coverage_history[-2] if len(coverage_history) > 1 else 0
                axes[row, col].set_title(f"Step {i + 1}: r={current_radius}, Cov {info['coverage']:.1%} (+{coverage_improvement:.1%})")
                axes[row, col].axis("off")
                
                step_idx += 1
            
            if done:
                break
        
        # Final result
        axes[1, 0].imshow(test_env.original_map, cmap="hot", alpha=0.6)
        for j, (x, y, r) in enumerate(test_env.placed_circles):
            if r >= 15:
                color = 'red'
            elif r >= 8:
                color = 'blue'
            else:
                color = 'green'
            circle = plt.Circle((y, x), r, fill=False, color=color, linewidth=2)
            axes[1, 0].add_patch(circle)
        
        norm_coverage = info["coverage"] / info["max_theoretical_coverage"]
        axes[1, 0].set_title(f"Final: {info['coverage']:.1%} ({norm_coverage:.1%} of max)")
        axes[1, 0].axis("off")
        
        # Coverage progress
        axes[1, 1].plot(coverage_history, 'g-', linewidth=2, marker='o')
        axes[1, 1].set_title("Coverage Progress")
        axes[1, 1].set_xlabel("Circle Placement")
        axes[1, 1].set_ylabel("Coverage")
        axes[1, 1].grid(True)
        
        # Configuration info
        config_text = f"Episode: {episode}\n"
        config_text += f"Circles: {len(test_env.radii)}\n"
        config_text += f"Radii: {test_env.radii}\n"
        config_text += f"Max Theoretical: {info['max_theoretical_coverage']:.1%}\n"
        config_text += f"Achieved: {info['coverage']:.1%}\n"
        config_text += f"Efficiency: {norm_coverage:.1%}\n"
        config_text += f"Training Steps: {self.training_step:,}\n"
        config_text += f"Buffer: {len(self.replay_buffer):,}"
        
        axes[1, 2].text(0.05, 0.95, config_text, fontsize=10, transform=axes[1, 2].transAxes, 
                       verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title("Configuration & Stats")
        axes[1, 2].axis("off")
        
        # Add legend for circle colors
        legend_text = "Circle Sizes:\n🔴 Large (≥15)\n🔵 Medium (8-14)\n🟢 Small (<8)"
        axes[0, 1].text(0.02, 0.02, legend_text, fontsize=10, transform=axes[0, 1].transAxes,
                       verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.suptitle(f"Randomized Radii Agent - Episode {episode} ({len(test_env.radii)} circles)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _quick_visualize(self, episode: int):
        """Quick visualization every 100 episodes."""
        try:
            self.visualize_strategy(episode, f"randomized_strategy_ep{episode}.png")
            print(f"📸 Visualization saved: randomized_strategy_ep{episode}.png")
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def _evaluate_and_visualize(self, episode: int):
        """Evaluation with randomized radii metrics."""
        print(f"\n{'=' * 80}")
        print(f"Episode {episode:,}/{self.config.n_episodes:,} - RANDOMIZED RADII")
        print(f"{'=' * 80}")
        
        # Performance stats
        recent_coverage = np.mean(self.episode_coverage[-1000:]) if len(self.episode_coverage) >= 1000 else np.mean(self.episode_coverage)
        recent_norm_coverage = np.mean(self.normalized_coverage[-1000:]) if len(self.normalized_coverage) >= 1000 else np.mean(self.normalized_coverage)
        recent_rewards = np.mean(self.episode_rewards[-1000:]) if len(self.episode_rewards) >= 1000 else np.mean(self.episode_rewards)
        recent_cov_imp = np.mean(self.avg_coverage_improvements[-1000:]) if len(self.avg_coverage_improvements) >= 1000 else np.mean(self.avg_coverage_improvements)
        best_coverage = max(self.episode_coverage)
        best_norm_coverage = max(self.normalized_coverage)
        
        print(f"\nPERFORMANCE:")
        print(f"  Recent avg coverage: {recent_coverage:.1%}")
        print(f"  Recent avg normalized coverage: {recent_norm_coverage:.1%}")
        print(f"  Best coverage: {best_coverage:.1%}")
        print(f"  Best normalized coverage: {best_norm_coverage:.1%}")
        print(f"  Recent avg reward: {recent_rewards:.2f}")
        print(f"  Recent avg coverage improvement: {recent_cov_imp:.3f}")
        print(f"  Training steps: {self.training_step:,}")
        print(f"  Buffer size: {len(self.replay_buffer):,}")
        
        # Configuration analysis
        if self.n_circles_history:
            recent_n_circles = self.n_circles_history[-1000:] if len(self.n_circles_history) >= 1000 else self.n_circles_history
            recent_max_theoretical = self.max_theoretical_coverage_history[-1000:] if len(self.max_theoretical_coverage_history) >= 1000 else self.max_theoretical_coverage_history
            
            print(f"\nCONFIGURATION ANALYSIS:")
            print(f"  Avg circles per episode: {np.mean(recent_n_circles):.1f} ± {np.std(recent_n_circles):.1f}")
            print(f"  Circle count range: {min(recent_n_circles)} - {max(recent_n_circles)}")
            print(f"  Avg theoretical max coverage: {np.mean(recent_max_theoretical):.1%}")
            print(f"  Theoretical coverage range: {min(recent_max_theoretical):.1%} - {max(recent_max_theoretical):.1%}")
        
        # Reward-coverage alignment
        if len(self.episode_rewards) > 100:
            correlation = np.corrcoef(self.episode_rewards[-1000:], self.normalized_coverage[-1000:])[0, 1]
            print(f"\nALIGNMENT:")
            print(f"  Reward-NormalizedCoverage correlation: {correlation:.3f}")
            if correlation > 0.7:
                print(f"  ✅ Strong positive correlation - rewards aligned with normalized coverage!")
            elif correlation > 0.3:
                print(f"  ⚠️  Moderate correlation - some alignment")
            else:
                print(f"  ❌ Weak correlation - rewards not well aligned")
        
        if self.losses:
            print(f"\nTRAINING:")
            print(f"  Recent avg loss: {np.mean(self.losses[-1000:]):.4f}")
            with self.epsilon_value.get_lock():
                print(f"  Epsilon: {self.epsilon_value.value:.3f}")
        
        # Save checkpoint
        if episode % self.config.save_every == 0:
            torch.save({
                "model_state_dict": self.agent.q_network.state_dict(),
                "target_state_dict": self.agent.target_network.state_dict(),
                "episode": episode,
                "training_step": self.training_step,
                "coverage_stats": {
                    "recent": recent_coverage,
                    "recent_normalized": recent_norm_coverage,
                    "best": best_coverage,
                    "best_normalized": best_norm_coverage,
                    "correlation": correlation if len(self.episode_rewards) > 100 else 0,
                }
            }, f"randomized_radii_model_ep{episode}.pth")
            print(f"Checkpoint saved: randomized_radii_model_ep{episode}.pth")
    
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
            "normalized_coverage": self.normalized_coverage,
            "coverage_improvements": self.coverage_improvements,
            "avg_coverage_improvements": self.avg_coverage_improvements,
            "radii_configs": self.radii_configs,
            "n_circles_history": self.n_circles_history,
            "max_theoretical_coverage_history": self.max_theoretical_coverage_history,
            "losses": self.losses,
            "config": self.config,
        }, "randomized_radii_final_model.pth")
        
        print("Final model saved: randomized_radii_final_model.pth")
    
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
    """Main randomized radii training function."""
    try:
        n_cores = mp.cpu_count()
    except:
        n_cores = 32
    
    # Optimized worker count
    n_workers = min(64, max(16, n_cores - 8))
    
    print(f"System: {n_cores} cores")
    print(f"Using {n_workers} workers for randomized radii training")
    
    config = RandomizedRadiiConfig(
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
    
    trainer = RandomizedRadiiTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()