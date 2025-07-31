"""Advanced enhancements for breaking through coverage plateaus."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import label, maximum_filter
import cv2


class AdvancedFeatureExtractor:
    """Enhanced feature extraction with more sophisticated analysis."""
    
    def __init__(self, map_size: int):
        self.map_size = map_size
        
    def extract_advanced_features(self, heatmap: np.ndarray, radius: float, 
                                  placed_circles: List[Tuple[float, float, float]]) -> Dict:
        """Extract advanced features for better decision making."""
        features = {}
        
        # 1. Multi-scale analysis
        features['multi_scale'] = self._multi_scale_analysis(heatmap, radius)
        
        # 2. Packing efficiency metrics
        features['packing'] = self._packing_efficiency_analysis(heatmap, radius, placed_circles)
        
        # 3. Connectivity analysis
        features['connectivity'] = self._connectivity_analysis(heatmap, radius)
        
        # 4. Optimal packing patterns
        features['patterns'] = self._pattern_analysis(heatmap, radius, placed_circles)
        
        # 5. Edge and corner utilization
        features['edge_corner'] = self._edge_corner_analysis(heatmap, radius)
        
        # 6. Void analysis
        features['voids'] = self._void_analysis(heatmap, radius)
        
        return features
    
    def _multi_scale_analysis(self, heatmap: np.ndarray, radius: float) -> Dict:
        """Analyze heatmap at multiple scales."""
        scales = [0.5, 1.0, 1.5, 2.0]
        multi_scale_features = {}
        
        for scale in scales:
            scaled_radius = int(radius * scale)
            if scaled_radius < 1:
                continue
                
            # Apply Gaussian blur to simulate different scales
            blurred = cv2.GaussianBlur(heatmap, (scaled_radius*2+1, scaled_radius*2+1), 0)
            
            # Find peaks at this scale
            peaks = self._find_peaks(blurred, scaled_radius)
            multi_scale_features[f'scale_{scale}'] = {
                'n_peaks': len(peaks),
                'peak_values': [blurred[p[0], p[1]] for p in peaks[:5]],
                'avg_value': blurred.mean()
            }
        
        return multi_scale_features
    
    def _packing_efficiency_analysis(self, heatmap: np.ndarray, radius: float, 
                                     placed_circles: List[Tuple[float, float, float]]) -> Dict:
        """Analyze packing efficiency and suggest improvements."""
        # Calculate current packing density
        total_area = self.map_size * self.map_size
        placed_area = sum(np.pi * r * r for _, _, r in placed_circles)
        current_density = placed_area / total_area
        
        # Estimate theoretical maximum packing for given radii
        if len(placed_circles) > 0:
            avg_radius = np.mean([r for _, _, r in placed_circles])
            # Hexagonal packing density ≈ 0.9069
            theoretical_max = 0.9069 * (placed_area / (np.pi * avg_radius * avg_radius)) * (np.pi * avg_radius * avg_radius) / total_area
        else:
            theoretical_max = 0.9069
        
        # Find gaps that could fit circles
        gap_mask = heatmap > 0
        gaps = self._find_circular_gaps(gap_mask, radius)
        
        return {
            'current_density': current_density,
            'theoretical_max': theoretical_max,
            'efficiency_ratio': current_density / theoretical_max if theoretical_max > 0 else 0,
            'n_viable_gaps': len(gaps),
            'gap_locations': gaps[:10]  # Top 10 gaps
        }
    
    def _connectivity_analysis(self, heatmap: np.ndarray, radius: float) -> Dict:
        """Analyze connectivity of high-value regions."""
        threshold = heatmap.max() * 0.5
        high_value_mask = heatmap > threshold
        
        # Find connected components
        labeled_array, num_components = label(high_value_mask)
        
        components = []
        for i in range(1, min(num_components + 1, 10)):  # Analyze top 10 components
            component_mask = labeled_array == i
            component_size = component_mask.sum()
            
            # Can this component fit a circle?
            can_fit = component_size >= np.pi * radius * radius * 0.7
            
            # Find best position within component
            if can_fit:
                best_pos = self._find_best_position_in_region(heatmap * component_mask, radius)
            else:
                best_pos = None
                
            components.append({
                'size': component_size,
                'can_fit': can_fit,
                'best_position': best_pos,
                'avg_value': heatmap[component_mask].mean() if component_size > 0 else 0
            })
        
        return {
            'num_components': num_components,
            'components': components,
            'fragmentation': 1.0 - (1.0 / max(num_components, 1))  # Higher = more fragmented
        }
    
    def _pattern_analysis(self, heatmap: np.ndarray, radius: float, 
                          placed_circles: List[Tuple[float, float, float]]) -> Dict:
        """Analyze placement patterns for optimization opportunities."""
        if len(placed_circles) < 2:
            return {'pattern_type': 'none', 'suggestions': []}
        
        # Analyze distances between placed circles
        distances = []
        for i, (x1, y1, r1) in enumerate(placed_circles):
            for x2, y2, r2 in placed_circles[i+1:]:
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                optimal_dist = r1 + r2  # Touching distance
                distances.append({
                    'actual': dist,
                    'optimal': optimal_dist,
                    'efficiency': optimal_dist / dist if dist > 0 else 0
                })
        
        avg_efficiency = np.mean([d['efficiency'] for d in distances]) if distances else 0
        
        # Detect pattern type
        if avg_efficiency > 0.9:
            pattern_type = 'tight_packing'
        elif avg_efficiency > 0.7:
            pattern_type = 'good_packing'
        else:
            pattern_type = 'loose_packing'
        
        # Suggest improvements
        suggestions = []
        if pattern_type == 'loose_packing':
            # Find positions that would create tighter packing
            for x in range(int(radius), int(self.map_size - radius), int(radius)):
                for y in range(int(radius), int(self.map_size - radius), int(radius)):
                    if self._would_create_tight_packing(x, y, radius, placed_circles):
                        value = self._calculate_circle_value(heatmap, x, y, radius)
                        if value > 0:
                            suggestions.append((x, y, value))
        
        suggestions.sort(key=lambda x: x[2], reverse=True)
        
        return {
            'pattern_type': pattern_type,
            'avg_packing_efficiency': avg_efficiency,
            'suggestions': suggestions[:5]
        }
    
    def _edge_corner_analysis(self, heatmap: np.ndarray, radius: float) -> Dict:
        """Analyze edge and corner utilization opportunities."""
        edge_width = int(radius * 1.5)
        
        # Define edge regions
        top_edge = heatmap[:edge_width, :]
        bottom_edge = heatmap[-edge_width:, :]
        left_edge = heatmap[:, :edge_width]
        right_edge = heatmap[:, -edge_width:]
        
        # Define corners
        corner_size = int(radius * 2)
        corners = {
            'top_left': heatmap[:corner_size, :corner_size],
            'top_right': heatmap[:corner_size, -corner_size:],
            'bottom_left': heatmap[-corner_size:, :corner_size],
            'bottom_right': heatmap[-corner_size:, -corner_size:]
        }
        
        # Calculate utilization potential
        edge_potential = {
            'top': top_edge.sum() / top_edge.size,
            'bottom': bottom_edge.sum() / bottom_edge.size,
            'left': left_edge.sum() / left_edge.size,
            'right': right_edge.sum() / right_edge.size
        }
        
        corner_potential = {
            name: region.sum() / region.size 
            for name, region in corners.items()
        }
        
        return {
            'edge_potential': edge_potential,
            'corner_potential': corner_potential,
            'best_edge': max(edge_potential, key=edge_potential.get),
            'best_corner': max(corner_potential, key=corner_potential.get)
        }
    
    def _void_analysis(self, heatmap: np.ndarray, radius: float) -> Dict:
        """Analyze voids and gaps in coverage."""
        # Create binary mask of uncovered areas
        uncovered = heatmap > heatmap.max() * 0.1
        
        # Find voids (connected uncovered regions)
        void_labels, n_voids = label(uncovered)
        
        voids = []
        for i in range(1, min(n_voids + 1, 20)):
            void_mask = void_labels == i
            void_size = void_mask.sum()
            
            if void_size > 0:
                # Find void center
                coords = np.argwhere(void_mask)
                center = coords.mean(axis=0).astype(int)
                
                # Calculate shape metrics
                if len(coords) > 0:
                    distances = np.sqrt(((coords - center) ** 2).sum(axis=1))
                    max_dist = distances.max()
                    avg_dist = distances.mean()
                    circularity = avg_dist / max_dist if max_dist > 0 else 0
                else:
                    circularity = 0
                
                # Can fit circle?
                can_fit = void_size >= np.pi * radius * radius * 0.5
                
                voids.append({
                    'size': void_size,
                    'center': center,
                    'circularity': circularity,
                    'can_fit': can_fit,
                    'value': heatmap[void_mask].sum()
                })
        
        # Sort by value potential
        voids.sort(key=lambda x: x['value'], reverse=True)
        
        return {
            'n_voids': n_voids,
            'total_void_area': uncovered.sum(),
            'voids': voids[:10],
            'fragmentation_score': n_voids / max(uncovered.sum() / 100, 1)  # More voids = more fragmented
        }
    
    def _find_peaks(self, heatmap: np.ndarray, radius: int) -> List[Tuple[int, int]]:
        """Find local peaks in heatmap."""
        # Use maximum filter to find local maxima
        local_max = maximum_filter(heatmap, size=2*radius+1)
        peaks = (heatmap == local_max) & (heatmap > heatmap.max() * 0.3)
        
        # Get peak coordinates
        peak_coords = np.argwhere(peaks)
        
        # Sort by value
        peak_values = [heatmap[p[0], p[1]] for p in peak_coords]
        sorted_indices = np.argsort(peak_values)[::-1]
        
        return [tuple(peak_coords[i]) for i in sorted_indices]
    
    def _find_circular_gaps(self, mask: np.ndarray, radius: float) -> List[Tuple[int, int]]:
        """Find gaps that could fit circles."""
        gaps = []
        
        # Use distance transform to find largest inscribed circles
        dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        
        # Find points that could be circle centers
        potential_centers = np.argwhere(dist_transform >= radius)
        
        for center in potential_centers:
            gaps.append(tuple(center))
        
        return gaps
    
    def _find_best_position_in_region(self, region: np.ndarray, radius: float) -> Optional[Tuple[int, int]]:
        """Find best position for circle within a region."""
        # Use convolution to find best position
        kernel_size = int(2 * radius + 1)
        kernel = np.zeros((kernel_size, kernel_size))
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                    kernel[i, j] = 1
        
        # Convolve to find best position
        conv_result = cv2.filter2D(region, -1, kernel)
        
        # Find maximum
        max_pos = np.unravel_index(conv_result.argmax(), conv_result.shape)
        
        if conv_result[max_pos] > 0:
            return max_pos
        return None
    
    def _would_create_tight_packing(self, x: float, y: float, radius: float, 
                                    placed_circles: List[Tuple[float, float, float]]) -> bool:
        """Check if placing circle at (x, y) would create tight packing."""
        if not placed_circles:
            return False
        
        # Count how many circles this would touch
        touching_count = 0
        for cx, cy, cr in placed_circles:
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if abs(dist - (radius + cr)) < 2:  # Within 2 pixels of touching
                touching_count += 1
        
        return touching_count >= 2  # Touches at least 2 circles
    
    def _calculate_circle_value(self, heatmap: np.ndarray, x: float, y: float, radius: float) -> float:
        """Calculate the value a circle would collect at position (x, y)."""
        value = 0
        for i in range(max(0, int(x - radius)), min(heatmap.shape[0], int(x + radius + 1))):
            for j in range(max(0, int(y - radius)), min(heatmap.shape[1], int(y + radius + 1))):
                if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                    value += heatmap[i, j]
        return value


class ImprovedCirclePlacementNet(nn.Module):
    """Enhanced neural network with attention mechanisms and deeper architecture."""
    
    def __init__(self, map_size=64, hidden_size=1024, n_heads=8):
        super(ImprovedCirclePlacementNet, self).__init__()
        self.map_size = map_size
        
        # Enhanced CNN backbone with residual connections
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(128, 128)
        self.res_block2 = self._make_residual_block(256, 256)
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Calculate CNN output size
        conv_out_size = (map_size // 4) * (map_size // 4) * 512
        
        # Enhanced feature size (including advanced features)
        feature_size = 100  # Increased from 50 to include advanced features
        
        # Multi-head attention for feature fusion
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Deeper fully connected layers
        self.fc1 = nn.Linear(conv_out_size + feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, map_size * map_size)
        
        # Enhanced normalization
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(0.3)
        
    def _make_residual_block(self, in_channels, out_channels):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, state_dict):
        """Forward pass with attention mechanisms."""
        # Unpack state
        current_map = state_dict["current_map"]
        placed_mask = state_dict["placed_mask"]
        value_density = state_dict["value_density"]
        features = state_dict["features"]
        
        batch_size = current_map.shape[0]
        
        # Stack channels
        x = torch.stack([current_map, placed_mask, value_density], dim=1)
        
        # CNN with residual connections
        x = F.relu(self.batch_norm1(self.conv1(x)))
        
        x = F.relu(self.batch_norm2(self.conv2(x)))
        residual = x
        x = self.res_block1(x)
        x = F.relu(x + residual)
        
        x = F.relu(self.batch_norm3(self.conv3(x)))
        residual = F.interpolate(x, scale_factor=1, mode='nearest')
        x = self.res_block2(x)
        x = F.relu(x + residual)
        
        x = F.relu(self.batch_norm4(self.conv4(x)))
        
        # Apply spatial attention
        attention_weights = self.spatial_attention(x)
        x = x * attention_weights
        
        # Flatten CNN output
        x_cnn = x.view(batch_size, -1)
        
        # Concatenate with features
        x = torch.cat([x_cnn, features], dim=1)
        
        # Deep fully connected layers with attention
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        
        # Self-attention on features
        x_attn = x.unsqueeze(1)  # Add sequence dimension
        x_attn, _ = self.feature_attention(x_attn, x_attn, x_attn)
        x = x + x_attn.squeeze(1)  # Residual connection
        
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = self.fc4(x)
        
        # Reshape to map size
        return x.view(batch_size, self.map_size, self.map_size)


class CurriculumLearningScheduler:
    """Curriculum learning to gradually increase problem difficulty."""
    
    def __init__(self, start_difficulty=0.3, end_difficulty=1.0, warmup_episodes=10000):
        self.start_difficulty = start_difficulty
        self.end_difficulty = end_difficulty
        self.warmup_episodes = warmup_episodes
        self.current_episode = 0
        
    def get_difficulty(self) -> float:
        """Get current difficulty level."""
        if self.current_episode >= self.warmup_episodes:
            return self.end_difficulty
        
        # Linear increase
        progress = self.current_episode / self.warmup_episodes
        return self.start_difficulty + (self.end_difficulty - self.start_difficulty) * progress
    
    def step(self):
        """Increment episode counter."""
        self.current_episode += 1
    
    def adjust_problem(self, base_radii: List[float], base_map_complexity: float) -> Tuple[List[float], float]:
        """Adjust problem difficulty based on current level."""
        difficulty = self.get_difficulty()
        
        # Adjust number of circles
        n_circles = int(len(base_radii) * difficulty)
        n_circles = max(3, n_circles)  # At least 3 circles
        
        # Adjust radii (larger circles are harder to pack)
        adjusted_radii = base_radii[:n_circles]
        if difficulty > 0.7:
            # Make some circles larger
            for i in range(min(3, len(adjusted_radii))):
                adjusted_radii[i] = min(adjusted_radii[i] * 1.2, 20)
        
        # Adjust map complexity
        adjusted_complexity = base_map_complexity * difficulty
        
        return adjusted_radii, adjusted_complexity


class ExperiencePrioritization:
    """Prioritize important experiences for replay."""
    
    def __init__(self, alpha=0.6, beta=0.4):
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.priorities = []
        self.max_priority = 1.0
        
    def add_experience(self, td_error: float, coverage_improvement: float):
        """Add experience with priority based on TD error and coverage improvement."""
        # Combine TD error and coverage improvement for priority
        priority = (abs(td_error) + 0.1) ** self.alpha
        priority *= (1 + coverage_improvement) ** 0.5  # Boost priority for good coverage improvements
        
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
        
        # Keep buffer size manageable
        if len(self.priorities) > 100000:
            self.priorities.pop(0)
    
    def sample_indices(self, batch_size: int, buffer_size: int) -> List[int]:
        """Sample indices based on priorities."""
        if len(self.priorities) < batch_size:
            return list(range(len(self.priorities)))
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities[-buffer_size:])
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(
            len(priorities),
            size=batch_size,
            p=probabilities,
            replace=False
        )
        
        return indices.tolist()
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < len(self.priorities):
                priority = (abs(td_error) + 0.1) ** self.alpha
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)


def create_enhanced_training_config():
    """Create configuration for enhanced training."""
    return {
        'network': 'improved',  # Use ImprovedCirclePlacementNet
        'use_curriculum': True,
        'use_prioritized_replay': True,
        'use_advanced_features': True,
        'batch_size': 64,  # Larger batch size
        'learning_rate': 5e-5,  # Lower learning rate for stability
        'gamma': 0.99,
        'tau': 0.005,  # Slightly higher for faster target updates
        'epsilon_start': 0.3,  # Lower starting epsilon due to better exploration
        'epsilon_end': 0.01,
        'epsilon_decay_steps': 50000,
        'buffer_size': 200000,  # Larger replay buffer
        'target_update_freq': 500,  # More frequent target updates
        'n_workers': 8,  # More parallel workers
        'episodes': 200000,  # More training episodes
        'save_freq': 1000,
        'eval_freq': 500,
        'warmup_episodes': 10000,  # Curriculum learning warmup
        'use_mixed_precision': True,
        'gradient_clip': 1.0,
        'reward_scaling': 2.0,  # Scale rewards for better learning
        'coverage_bonus_weight': 0.5,  # Additional weight for coverage improvements
        'exploration_bonus': 0.1,  # Bonus for exploring new configurations
    }