import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import math
from numba import njit

try:
    from scipy.ndimage import label, maximum_filter, gaussian_filter
except ImportError:
    print("Warning: scipy not available. Some features will be limited.")
    label = None
    maximum_filter = None
    gaussian_filter = None


class HeatmapFeatureExtractor:
    """Extract human-interpretable features from the heatmap."""

    def __init__(self, map_size=64):
        self.map_size = map_size

    def extract_features(self, heatmap, radius):
        """Extract strategic features that humans would consider."""
        features = {}

        # 1. Find high-value regions
        threshold = heatmap.max() * 0.7  # Top 30% values
        high_value_mask = heatmap > threshold

        # 2. Identify connected components (clusters)
        if label is not None:
            labeled_array, num_clusters = label(high_value_mask)
        else:
            # Fallback: treat all high-value areas as one cluster
            labeled_array = high_value_mask.astype(int)
            num_clusters = 1 if np.any(high_value_mask) else 0

        # 3. Calculate cluster properties
        clusters = []
        for i in range(1, num_clusters + 1):
            cluster_mask = labeled_array == i
            cluster_positions = np.argwhere(cluster_mask)
            if len(cluster_positions) > 0:
                # Cluster center
                center = cluster_positions.mean(axis=0)
                # Cluster size
                size = len(cluster_positions)
                # Average density
                density = heatmap[cluster_mask].mean()
                # Can fit circle?
                can_fit = size >= np.pi * radius * radius * 0.7

                clusters.append(
                    {
                        "center": center,
                        "size": size,
                        "density": density,
                        "can_fit": can_fit,
                        "positions": cluster_positions,
                    }
                )

        # Sort clusters by value potential
        clusters.sort(key=lambda x: x["density"] * x["size"], reverse=True)

        features["clusters"] = clusters
        features["num_clusters"] = len(clusters)

        # 4. Calculate coverage potential for current radius
        if maximum_filter is not None:
            # Use maximum filter to find best positions
            footprint = np.zeros((2 * radius + 1, 2 * radius + 1))
            for i in range(2 * radius + 1):
                for j in range(2 * radius + 1):
                    if (i - radius) ** 2 + (j - radius) ** 2 <= radius**2:
                        footprint[i, j] = 1

            potential_values = maximum_filter(
                heatmap, footprint=footprint, mode="constant"
            )
            features["best_positions"] = np.argwhere(
                potential_values == potential_values.max()
            )
            features["max_potential"] = potential_values.max()
        else:
            # Fallback: find positions with highest local average
            best_val = 0
            best_positions = []
            for x in range(radius, heatmap.shape[0] - radius):
                for y in range(radius, heatmap.shape[1] - radius):
                    local_sum = 0
                    for i in range(
                        max(0, x - radius), min(heatmap.shape[0], x + radius + 1)
                    ):
                        for j in range(
                            max(0, y - radius), min(heatmap.shape[1], y + radius + 1)
                        ):
                            if (i - x) ** 2 + (j - y) ** 2 <= radius**2:
                                local_sum += heatmap[i, j]
                    if local_sum > best_val:
                        best_val = local_sum
                        best_positions = [(x, y)]
                    elif local_sum == best_val:
                        best_positions.append((x, y))

            features["best_positions"] = (
                np.array(best_positions)
                if best_positions
                else np.array([[radius, radius]])
            )
            features["max_potential"] = best_val

        return features


class SmartCirclePlacementNet(nn.Module):
    """Network that incorporates spatial features and heuristics."""

    def __init__(self, map_size=64, hidden_size=512):
        super(SmartCirclePlacementNet, self).__init__()
        self.map_size = map_size

        # CNN for spatial understanding
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Calculate CNN output size
        conv_out_size = (map_size // 4) * (map_size // 4) * 256

        # Additional features: radius, progress, cluster info
        feature_size = 10  # radius, progress, num_clusters, max_density, etc.

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size + feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, map_size * map_size)

        # Normalization
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(0.2)

    def forward(self, state_dict):
        """Forward pass with rich state representation."""
        # Unpack state
        current_map = state_dict["current_map"]
        placed_mask = state_dict["placed_mask"]
        value_density = state_dict["value_density"]
        features = state_dict["features"]

        batch_size = current_map.shape[0]

        # Stack channels: current values, placed mask, value density
        x = torch.stack([current_map, placed_mask, value_density], dim=1)

        # CNN layers
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.batch_norm3(self.conv3(x)))

        # Flatten CNN output
        x = x.view(batch_size, -1)

        # Concatenate with additional features
        x = torch.cat([x, features], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        # Reshape to map size
        return x.view(batch_size, self.map_size, self.map_size)


class AdvancedCirclePlacementEnv:
    """Enhanced environment with better state representation."""

    def __init__(self, map_size=64, radii=None):
        self.map_size = map_size
        self.radii = (
            radii if radii is not None else [20, 17, 14, 12, 12, 8, 7, 6, 5, 4, 3, 2, 1]
        )
        self.feature_extractor = HeatmapFeatureExtractor(map_size)
        self.reset()

    def reset(self, weighted_matrix=None):
        """Reset with enhanced state tracking."""
        if weighted_matrix is None:
            self.original_map = random_seeder(self.map_size, time_steps=100000)
        else:
            self.original_map = weighted_matrix.copy()

        self.current_map = self.original_map.copy()
        self.current_radius_idx = 0
        self.placed_circles = []
        self.total_weight_collected = 0

        # Additional tracking
        self.placement_order = []  # Track which radius was placed when
        self.coverage_history = []  # Track coverage over time

        return self._get_enhanced_state()

    def _get_enhanced_state(self):
        """Get rich state representation with human-like features."""
        # Handle case when all circles are placed
        if self.current_radius_idx >= len(self.radii):
            return None

        radius = self.radii[self.current_radius_idx]

        # Extract strategic features
        features = self.feature_extractor.extract_features(self.current_map, radius)

        # Create state dictionary
        state_dict = {
            "current_map": self.current_map,
            "placed_mask": self._get_placed_mask(),
            "value_density": self._get_value_density_map(),
            "features": self._encode_features(features, radius),
            "raw_features": features,  # For visualization
        }

        return state_dict

    def _get_placed_mask(self):
        """Create a mask showing where circles are placed."""
        mask = np.zeros((self.map_size, self.map_size))
        for x, y, r in self.placed_circles:
            for i in range(max(0, int(x - r)), min(self.map_size, int(x + r + 1))):
                for j in range(max(0, int(y - r)), min(self.map_size, int(y + r + 1))):
                    if (i - x) ** 2 + (j - y) ** 2 <= r**2:
                        mask[i, j] = 1.0
        return mask

    def _get_value_density_map(self):
        """Create a smoothed value density map."""
        # Apply gaussian smoothing to highlight high-value regions
        if gaussian_filter is not None:
            density = gaussian_filter(self.current_map, sigma=3.0)
        else:
            # Simple averaging as fallback
            density = self.current_map.copy()

        if density.max() > 0:
            density = density / density.max()
        return density

    def _encode_features(self, features, radius):
        """Encode features as a vector."""
        encoded = np.zeros(10)

        # Current radius (normalized)
        encoded[0] = radius / max(self.radii)

        # Progress
        encoded[1] = self.current_radius_idx / len(self.radii)

        # Number of high-value clusters
        encoded[2] = min(features["num_clusters"] / 10, 1.0)

        # Maximum potential value
        if self.original_map.max() > 0:
            encoded[3] = features["max_potential"] / self.original_map.max()

        # Coverage so far
        if self.original_map.sum() > 0:
            encoded[4] = self.total_weight_collected / self.original_map.sum()

        # Cluster fit information
        can_fit_count = sum(1 for c in features["clusters"] if c["can_fit"])
        encoded[5] = min(can_fit_count / 5, 1.0)

        # Average cluster density
        if features["clusters"]:
            avg_density = np.mean([c["density"] for c in features["clusters"]])
            if self.original_map.max() > 0:
                encoded[6] = avg_density / self.original_map.max()

        # Remaining circles
        encoded[7] = (len(self.radii) - self.current_radius_idx) / len(self.radii)

        # Largest remaining radius
        if self.current_radius_idx < len(self.radii) - 1:
            encoded[8] = self.radii[self.current_radius_idx + 1] / max(self.radii)

        # Space utilization
        placed_area = sum(np.pi * r * r for _, _, r in self.placed_circles)
        total_area = self.map_size * self.map_size
        encoded[9] = placed_area / total_area

        return encoded

    def get_suggested_positions(self, n_suggestions=5):
        """Get human-like position suggestions for current radius."""
        radius = self.radii[self.current_radius_idx]
        features = self.feature_extractor.extract_features(self.current_map, radius)

        suggestions = []

        # 1. Try cluster centers first
        for cluster in features["clusters"][:3]:  # Top 3 clusters
            if cluster["can_fit"]:
                center = cluster["center"].astype(int)
                if self._is_valid_position(center[0], center[1], radius):
                    suggestions.append(tuple(center))

        # 2. Try best potential positions
        for pos in features["best_positions"][:10]:
            if self._is_valid_position(pos[0], pos[1], radius):
                suggestions.append(tuple(pos))

        # Remove duplicates and limit
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)
                if len(unique_suggestions) >= n_suggestions:
                    break

        return unique_suggestions

    def _is_valid_position(self, x, y, radius):
        """Check if position is valid (no overlap, within bounds)."""
        # Boundary check
        if x < radius or x >= self.map_size - radius:
            return False
        if y < radius or y >= self.map_size - radius:
            return False

        # Overlap check
        for px, py, pr in self.placed_circles:
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if dist < radius + pr:
                return False

        return True

    def step(self, action):
        """Take a step with enhanced reward shaping."""
        x, y = action
        radius = self.radii[self.current_radius_idx]

        # Calculate base reward
        included_weight = compute_included(self.current_map, x, y, radius)

        # Calculate strategic bonuses
        reward = 0

        # 1. Base collection reward
        if included_weight > 0:
            reward = included_weight / (
                np.pi * radius * radius * self.original_map.max()
            )
            reward = np.clip(reward, 0, 1)

        # 2. Efficiency bonus - reward for using space well
        circle_area = np.pi * radius * radius
        actual_collected = included_weight
        max_possible = circle_area * self.original_map.max()
        efficiency = actual_collected / max(max_possible, 1)
        reward += 0.3 * (efficiency**2)

        # 3. Cluster completion bonus
        state = self._get_enhanced_state()
        features = state["raw_features"]
        for cluster in features["clusters"]:
            cluster_center = cluster["center"]
            dist_to_cluster = np.sqrt(
                (x - cluster_center[0]) ** 2 + (y - cluster_center[1]) ** 2
            )
            if dist_to_cluster < radius * 1.5:
                reward += 0.1
                break

        # 4. Strategic placement bonus for large circles
        if radius >= 8:  # Large circles
            # Reward placing in high-density areas
            local_density = self.current_map[
                max(0, x - radius) : min(self.map_size, x + radius + 1),
                max(0, y - radius) : min(self.map_size, y + radius + 1),
            ].mean()
            if self.original_map.max() > 0:
                density_ratio = local_density / self.original_map.max()
                reward += 0.2 * density_ratio

        # Update environment
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

        # Move to next radius
        self.current_radius_idx += 1
        done = self.current_radius_idx >= len(self.radii)

        # Calculate coverage
        coverage = 1 - (self.current_map.sum() / self.original_map.sum())
        self.coverage_history.append(coverage)

        info = {
            "coverage": coverage,
            "included_weight": included_weight,
            "efficiency": efficiency,
        }

        if done:
            # Final bonus for good overall coverage
            if coverage > 0.8:
                reward += 1.0
            elif coverage > 0.6:
                reward += 0.5

            # Return None for next state when done
            return None, reward, done, info

        return self._get_enhanced_state(), reward, done, info

    def get_valid_actions_mask(self):
        """Get mask with heuristic guidance."""
        radius = self.radii[self.current_radius_idx]
        mask = np.ones((self.map_size, self.map_size), dtype=bool)

        # Basic validity checks
        mask[: int(radius), :] = False
        mask[-int(radius) :, :] = False
        mask[:, : int(radius)] = False
        mask[:, -int(radius) :] = False

        # No overlap check
        for px, py, pr in self.placed_circles:
            min_distance = radius + pr
            for i in range(
                max(0, int(px - min_distance)),
                min(self.map_size, int(px + min_distance + 1)),
            ):
                for j in range(
                    max(0, int(py - min_distance)),
                    min(self.map_size, int(py + min_distance + 1)),
                ):
                    dist = np.sqrt((i - px) ** 2 + (j - py) ** 2)
                    if dist < min_distance:
                        mask[i, j] = False

        # Prefer high-value areas
        if np.any(mask & (self.current_map > self.current_map.max() * 0.3)):
            # If we have high-value options, strongly prefer them
            value_bonus = (self.current_map > self.current_map.max() * 0.3).astype(
                float
            )
            # Don't completely eliminate low-value areas, just deprioritize
            # This helps exploration
            return mask.astype(float) * (0.1 + 0.9 * value_bonus)

        return mask.astype(float)


# Copy the njit functions
@njit()
def random_seeder(dim, time_steps=100000):
    x = np.random.uniform(0, 1, (dim, dim))
    seed_pos_x = int(np.random.uniform(0, dim))
    seed_pos_y = int(np.random.uniform(0, dim))
    tele_prob = 0.001
    for i in range(time_steps):
        x[seed_pos_x, seed_pos_y] += np.random.uniform(0, 1)
        if np.random.uniform() < tele_prob:
            seed_pos_x = int(np.random.uniform(0, dim))
            seed_pos_y = int(np.random.uniform(0, dim))
        else:
            if np.random.uniform() < 0.5:
                seed_pos_x += 1
            if np.random.uniform() < 0.5:
                seed_pos_x += -1
            if np.random.uniform() < 0.5:
                seed_pos_y += 1
            if np.random.uniform() < 0.5:
                seed_pos_y += -1
            seed_pos_x = int(max(min(seed_pos_x, dim - 1), 0))
            seed_pos_y = int(max(min(seed_pos_y, dim - 1), 0))
    return x


@njit()
def compute_included(weighted_matrix, x, y, radius):
    included_weight = 0.0
    for i in range(
        max(0, int(x - radius)), min(weighted_matrix.shape[0], int(x + radius + 1))
    ):
        for j in range(
            max(0, int(y - radius)), min(weighted_matrix.shape[1], int(y + radius + 1))
        ):
            if (i - x) ** 2 + (j - y) ** 2 <= radius**2:
                included_weight += weighted_matrix[i, j]
                weighted_matrix[i, j] = 0
    return included_weight


class GuidedDQNAgent:
    """DQN agent with human-like heuristics and better exploration."""

    def __init__(
        self,
        map_size=64,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=0.5,
        epsilon_end=0.01,
        epsilon_decay_steps=2000,
        batch_size=32,
        buffer_size=50000,
        tau=0.001,
        use_suggestions=True,
        suggestion_prob=0.3,
    ):
        self.map_size = map_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_network = SmartCirclePlacementNet(map_size).to(self.device)
        self.target_network = SmartCirclePlacementNet(map_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.tau = tau
        self.steps_done = 0

        # Heuristic guidance
        self.use_suggestions = use_suggestions
        self.suggestion_prob = suggestion_prob

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

    def act(self, state_dict, env, valid_mask=None):
        """Choose action with optional heuristic guidance."""
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * self.steps_done / self.epsilon_decay_steps)

        # Sometimes use human-like suggestions
        if (
            self.use_suggestions
            and random.random() < self.suggestion_prob * self.epsilon
        ):
            suggestions = env.get_suggested_positions(n_suggestions=5)
            if suggestions:
                return random.choice(suggestions)

        # Epsilon-greedy with Q-network
        if random.random() < self.epsilon:
            # Random valid action
            if valid_mask is not None:
                valid_positions = np.argwhere(valid_mask > 0.5)
                if len(valid_positions) > 0:
                    # Weight by mask values for smarter random exploration
                    weights = valid_mask[valid_positions[:, 0], valid_positions[:, 1]]
                    weights = weights / weights.sum()
                    idx = np.random.choice(len(valid_positions), p=weights)
                    return tuple(valid_positions[idx])
            return (
                random.randint(0, self.map_size - 1),
                random.randint(0, self.map_size - 1),
            )

        # Use Q-network
        with torch.no_grad():
            # Prepare batch state
            state_batch = self._prepare_state_batch([state_dict])
            q_values = self.q_network(state_batch).squeeze(0)

            # Apply valid mask
            if valid_mask is not None:
                mask_tensor = torch.FloatTensor(valid_mask).to(self.device)
                q_values = (
                    q_values + (mask_tensor - 1) * 1e10
                )  # Large negative for invalid

            # Get best action
            action_idx = q_values.view(-1).argmax().item()
            return (action_idx // self.map_size, action_idx % self.map_size)

    def _prepare_state_batch(self, state_dicts):
        """Prepare batch of states for network."""
        batch_size = len(state_dicts)

        current_maps = torch.FloatTensor(
            np.array([s["current_map"] for s in state_dicts])
        ).to(self.device)

        placed_masks = torch.FloatTensor(
            np.array([s["placed_mask"] for s in state_dicts])
        ).to(self.device)

        value_densities = torch.FloatTensor(
            np.array([s["value_density"] for s in state_dicts])
        ).to(self.device)

        features = torch.FloatTensor(np.array([s["features"] for s in state_dicts])).to(
            self.device
        )

        return {
            "current_map": current_maps,
            "placed_mask": placed_masks,
            "value_density": value_densities,
            "features": features,
        }

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = [e[0] for e in batch]
        actions = [e[1] for e in batch]
        rewards = [e[2] for e in batch]
        next_states = [e[3] for e in batch if e[3] is not None]
        dones = [e[4] for e in batch]

        # Prepare batches
        state_batch = self._prepare_state_batch(states)

        # Current Q values
        current_q_values = self.q_network(state_batch)
        action_indices = torch.LongTensor(
            [[a[0] * self.map_size + a[1]] for a in actions]
        ).to(self.device)
        current_q_values = (
            current_q_values.view(self.batch_size, -1)
            .gather(1, action_indices)
            .squeeze()
        )

        # Next Q values
        next_q_values = torch.zeros(self.batch_size).to(self.device)
        if next_states:
            next_state_batch = self._prepare_state_batch(next_states)
            non_final_mask = torch.tensor([not d for d in dones], dtype=torch.bool).to(
                self.device
            )
            with torch.no_grad():
                # Double DQN
                next_actions = (
                    self.q_network(next_state_batch)
                    .view(len(next_states), -1)
                    .max(1)[1]
                )
                next_q_values[non_final_mask] = (
                    self.target_network(next_state_batch)
                    .view(len(next_states), -1)
                    .gather(1, next_actions.unsqueeze(1))
                    .squeeze()
                )

        # Compute targets
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        targets = rewards_tensor + self.gamma * next_q_values

        # Loss
        loss = F.smooth_l1_loss(current_q_values, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        return loss.item()
