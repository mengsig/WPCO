#!/usr/bin/env python3
"""Enhanced training script to break through coverage plateaus."""

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


class EnhancedCirclePlacementEnv(AdvancedCirclePlacementEnv):
    """Enhanced environment with advanced features."""
    
    def __init__(self, map_size=128, radii=None, use_advanced_features=True):
        # Set attributes before calling super().__init__
        self.use_advanced_features = use_advanced_features
        if use_advanced_features:
            self.advanced_extractor = AdvancedFeatureExtractor(map_size)
        
        # Now call parent init which will call reset()
        super().__init__(map_size, radii)
    
    def _get_enhanced_state(self):
        """Get state with advanced features."""
        state = super()._get_enhanced_state()
        
        if state is None or not self.use_advanced_features:
            return state
        
        # Extract advanced features
        radius = self.radii[self.current_radius_idx]
        advanced_features = self.advanced_extractor.extract_advanced_features(
            self.current_map, radius, self.placed_circles
        )
        
        # Encode advanced features into feature vector
        extended_features = self._encode_advanced_features(state['features'], advanced_features)
        state['features'] = extended_features
        state['advanced_features'] = advanced_features
        
        return state
    
    def _encode_advanced_features(self, base_features, advanced_features):
        """Encode advanced features into extended feature vector."""
        # Start with base features (50 dimensions)
        extended = np.zeros(100)  # Extend to 100 dimensions
        extended[:len(base_features)] = base_features
        
        idx = len(base_features)
        
        # Packing efficiency features
        packing = advanced_features['packing']
        extended[idx] = packing['current_density']
        extended[idx+1] = packing['efficiency_ratio']
        extended[idx+2] = min(packing['n_viable_gaps'] / 10, 1.0)
        idx += 3
        
        # Connectivity features
        connectivity = advanced_features['connectivity']
        extended[idx] = min(connectivity['num_components'] / 10, 1.0)
        extended[idx+1] = connectivity['fragmentation']
        idx += 2
        
        # Pattern features
        patterns = advanced_features['patterns']
        pattern_encoding = {
            'none': 0, 'loose_packing': 0.33, 
            'good_packing': 0.66, 'tight_packing': 1.0
        }
        extended[idx] = pattern_encoding.get(patterns['pattern_type'], 0)
        extended[idx+1] = patterns.get('avg_packing_efficiency', 0)
        idx += 2
        
        # Edge/corner features
        edge_corner = advanced_features['edge_corner']
        extended[idx] = max(edge_corner['edge_potential'].values())
        extended[idx+1] = max(edge_corner['corner_potential'].values())
        idx += 2
        
        # Void features
        voids = advanced_features['voids']
        extended[idx] = min(voids['n_voids'] / 20, 1.0)
        extended[idx+1] = min(voids['total_void_area'] / (self.map_size * self.map_size), 1.0)
        extended[idx+2] = voids['fragmentation_score']
        idx += 3
        
        # Multi-scale features (sample from scale 1.0)
        if 'scale_1.0' in advanced_features['multi_scale']:
            scale_features = advanced_features['multi_scale']['scale_1.0']
            extended[idx] = min(scale_features['n_peaks'] / 10, 1.0)
            extended[idx+1] = scale_features['avg_value'] / self.original_map.max() if self.original_map.max() > 0 else 0
        
        return extended
    
    def get_placement_suggestions(self):
        """Get advanced placement suggestions."""
        if not self.use_advanced_features or self.current_radius_idx >= len(self.radii):
            return []
        
        radius = self.radii[self.current_radius_idx]
        advanced_features = self.advanced_extractor.extract_advanced_features(
            self.current_map, radius, self.placed_circles
        )
        
        suggestions = []
        
        # Add pattern-based suggestions
        if advanced_features['patterns']['suggestions']:
            for x, y, value in advanced_features['patterns']['suggestions']:
                if self._is_valid_position(x, y, radius):
                    suggestions.append((x, y, value, 'pattern'))
        
        # Add void-filling suggestions
        for void in advanced_features['voids']['voids'][:3]:
            if void['can_fit']:
                x, y = void['center']
                if self._is_valid_position(x, y, radius):
                    suggestions.append((x, y, void['value'], 'void'))
        
        # Add connectivity-based suggestions
        for component in advanced_features['connectivity']['components'][:3]:
            if component['can_fit'] and component['best_position']:
                x, y = component['best_position']
                if self._is_valid_position(x, y, radius):
                    suggestions.append((x, y, component['avg_value'], 'connectivity'))
        
        # Sort by value
        suggestions.sort(key=lambda x: x[2], reverse=True)
        
        return suggestions[:5]


class EnhancedDQNAgent(GuidedDQNAgent):
    """Enhanced DQN agent with improved network and training."""
    
    def __init__(self, map_size=64, config=None):
        if config is None:
            config = create_enhanced_training_config()
        
        super().__init__(
            map_size=map_size,
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            epsilon_start=config['epsilon_start'],
            epsilon_end=config['epsilon_end'],
            epsilon_decay_steps=config['epsilon_decay_steps'],
            batch_size=config['batch_size'],
            buffer_size=config['buffer_size'],
            tau=config['tau'],
        )
        
        # Replace network with improved version
        if config.get('network') == 'improved':
            self.q_network = ImprovedCirclePlacementNet(map_size).to(self.device)
            self.target_network = ImprovedCirclePlacementNet(map_size).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Update optimizer
            self.optimizer = optim.Adam(
                self.q_network.parameters(), 
                lr=config['learning_rate'],
                weight_decay=1e-5
            )
        
        self.config = config
        
        # Experience prioritization
        if config.get('use_prioritized_replay'):
            self.prioritizer = ExperiencePrioritization()
        else:
            self.prioritizer = None
    
    def act_with_suggestions(self, state_dict, env):
        """Act with advanced suggestions."""
        # Get advanced suggestions
        if hasattr(env, 'get_placement_suggestions'):
            suggestions = env.get_placement_suggestions()
            
            # Use suggestions with higher probability early in training
            suggestion_prob = self.suggestion_prob * (1 + self.epsilon)
            
            if suggestions and random.random() < suggestion_prob:
                # Choose from suggestions based on their value
                values = [s[2] for s in suggestions]
                total_value = sum(values)
                if total_value > 0:
                    probs = [v / total_value for v in values]
                    idx = np.random.choice(len(suggestions), p=probs)
                    return (suggestions[idx][0], suggestions[idx][1])
        
        # Otherwise use standard action selection
        return self.act(state_dict, env)
    
    def enhanced_replay(self):
        """Enhanced replay with prioritized experience replay."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample based on priorities if available
        if self.prioritizer:
            indices = self.prioritizer.sample_indices(self.batch_size, len(self.memory))
            batch = [self.memory[i] for i in indices]
        else:
            batch = random.sample(self.memory, self.batch_size)
        
        # Standard replay logic
        states = [e[0] for e in batch]
        actions = [e[1] for e in batch]
        rewards = [e[2] for e in batch]
        next_states = [e[3] for e in batch]  # Keep all next states (including None)
        dones = [e[4] for e in batch]
        
        # Scale rewards if configured
        reward_scale = self.config.get('reward_scaling', 1.0)
        rewards = [r * reward_scale for r in rewards]
        
        # Prepare batches
        state_batch = self._prepare_state_batch(states)
        
        # Current Q values
        current_q_values = self.q_network(state_batch)
        
        # Ensure actions are within bounds and create indices
        action_indices = []
        for a in actions:
            x, y = int(a[0]), int(a[1])
            # Clamp to valid range
            x = max(0, min(x, self.map_size - 1))
            y = max(0, min(y, self.map_size - 1))
            idx = x * self.map_size + y
            action_indices.append([idx])
        
        action_indices = torch.LongTensor(action_indices).to(self.device)
        
        # Get Q values for taken actions
        if current_q_values.dim() == 3:
            current_q_values = current_q_values.view(len(batch), -1)
        
        current_q_values = current_q_values.gather(1, action_indices).squeeze(-1)
        
        # Next Q values
        next_q_values = torch.zeros(len(batch)).to(self.device)
        
        # Find which states are not terminal and have valid next states
        non_final_next_states = []
        non_final_indices = []
        for i, (next_state, done) in enumerate(zip(next_states, dones)):
            if not done and next_state is not None:
                non_final_next_states.append(next_state)
                non_final_indices.append(i)
        
        if len(non_final_next_states) > 0:
            # Prepare next state batch
            next_state_batch = self._prepare_state_batch(non_final_next_states)
            
            with torch.no_grad():
                # Get next Q values
                next_q_batch = self.q_network(next_state_batch)
                if next_q_batch.dim() == 3:
                    next_q_batch = next_q_batch.view(len(non_final_next_states), -1)
                
                # Double DQN: action selection from q_network, evaluation from target_network
                next_actions = next_q_batch.max(1)[1].unsqueeze(-1)
                
                target_q_batch = self.target_network(next_state_batch)
                if target_q_batch.dim() == 3:
                    target_q_batch = target_q_batch.view(len(non_final_next_states), -1)
                
                # Get Q values for next actions
                next_q_selected = target_q_batch.gather(1, next_actions).squeeze(-1)
                
                # Assign to appropriate indices
                for i, idx in enumerate(non_final_indices):
                    next_q_values[idx] = next_q_selected[i]
        
        # Compute targets
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        targets = rewards_tensor + self.gamma * next_q_values
        
        # Calculate TD errors for prioritization
        td_errors = (targets - current_q_values).detach().cpu().numpy()
        
        # Update priorities if using prioritized replay
        if self.prioritizer and hasattr(indices, '__iter__'):
            self.prioritizer.update_priorities(indices, td_errors)
        
        # Loss with gradient clipping
        loss = nn.functional.smooth_l1_loss(current_q_values, targets.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_clip = self.config.get('gradient_clip', 1.0)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), grad_clip)
        
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update_target_network()
        
        return loss.item()


def train_enhanced_agent(episodes=100000, debug=False):
    """Train agent with all enhancements."""
    print("=" * 80)
    print("ENHANCED TRAINING TO BREAK COVERAGE PLATEAU")
    print("=" * 80)
    
    # Configuration
    config = create_enhanced_training_config()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Environment and agent
    env = EnhancedCirclePlacementEnv(
        map_size=128,
        use_advanced_features=config['use_advanced_features']
    )
    
    agent = EnhancedDQNAgent(map_size=128, config=config)
    
    if debug:
        print(f"Environment map size: {env.map_size}")
        print(f"Agent map size: {agent.map_size}")
        print(f"Initial radii: {env.radii}")
    
    # Curriculum learning
    if config['use_curriculum']:
        curriculum = CurriculumLearningScheduler(
            warmup_episodes=config['warmup_episodes']
        )
    else:
        curriculum = None
    
    # Periodic checkers
    save_checker = RobustPeriodicChecker(config['save_freq'])
    eval_checker = RobustPeriodicChecker(config['eval_freq'])
    
    # Training metrics
    episode_rewards = []
    episode_coverage = []
    best_coverage = 0
    coverage_plateau_counter = 0
    
    # Training loop
    pbar = tqdm(range(episodes), desc="Training")
    
    for episode in pbar:
        # Adjust difficulty if using curriculum
        if curriculum:
            base_radii = [20, 17, 14, 12, 12, 8, 7, 6, 5, 4, 3, 2, 1]
            adjusted_radii, map_complexity = curriculum.adjust_problem(base_radii, 1.0)
            env.radii = adjusted_radii
            curriculum.step()
        
        # Reset environment
        state = env.reset()
        episode_reward = 0
        previous_coverage = 0
        
        # Episode loop
        while state is not None:
            # Choose action
            action = agent.act_with_suggestions(state, env)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Add coverage improvement bonus
            coverage_improvement = info['coverage'] - previous_coverage
            if coverage_improvement > 0:
                reward += config['coverage_bonus_weight'] * coverage_improvement
            
            # Add exploration bonus for new configurations
            if len(env.placed_circles) > 3:
                # Check if this creates a new pattern
                reward += config['exploration_bonus'] * np.random.random() * agent.epsilon
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Add to prioritizer if available
            if agent.prioritizer and len(agent.memory) > 1:
                # Calculate approximate TD error
                with torch.no_grad():
                    state_tensor = agent._prepare_state_batch([state])
                    q_value = agent.q_network(state_tensor).max().item()
                    next_q = 0
                    if next_state:
                        next_tensor = agent._prepare_state_batch([next_state])
                        next_q = agent.target_network(next_tensor).max().item()
                    td_error = reward + agent.gamma * next_q - q_value
                
                agent.prioritizer.add_experience(td_error, coverage_improvement)
            
            # Update state
            state = next_state
            episode_reward += reward
            previous_coverage = info['coverage']
            
            # Train
            if len(agent.memory) >= agent.batch_size:
                loss = agent.enhanced_replay()
            
            if done:
                break
        
        # Record metrics
        final_coverage = info['coverage']
        episode_rewards.append(episode_reward)
        episode_coverage.append(final_coverage)
        
        # Update best coverage
        if final_coverage > best_coverage:
            best_coverage = final_coverage
            coverage_plateau_counter = 0
        else:
            coverage_plateau_counter += 1
        
        # Update progress bar
        recent_coverage = np.mean(episode_coverage[-100:]) if episode_coverage else 0
        pbar.set_postfix({
            'Coverage': f'{recent_coverage:.1%}',
            'Best': f'{best_coverage:.1%}',
            'Reward': f'{np.mean(episode_rewards[-100:]):.1f}',
            'Epsilon': f'{agent.epsilon:.3f}'
        })
        
        # Periodic evaluation
        if eval_checker.should_execute(episode):
            print(f"\n[Episode {episode}] Coverage: {recent_coverage:.1%} (Best: {best_coverage:.1%})")
            
            # Check for plateau
            if coverage_plateau_counter > 1000:
                print("WARNING: Coverage plateau detected! Consider:")
                print("- Increasing exploration (epsilon)")
                print("- Adjusting reward weights")
                print("- Adding more training episodes")
        
        # Save checkpoint
        if save_checker.should_execute(episode):
            torch.save({
                'episode': episode,
                'model_state': agent.q_network.state_dict(),
                'optimizer_state': agent.optimizer.state_dict(),
                'best_coverage': best_coverage,
                'config': config
            }, f'enhanced_model_ep{episode}_cov{best_coverage:.0%}.pth')
    
    print(f"\nTraining complete! Best coverage: {best_coverage:.1%}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_coverage)
    plt.axhline(y=0.37, color='r', linestyle='--', label='Previous plateau')
    plt.title('Coverage Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Coverage')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_rewards)
    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    
    plt.tight_layout()
    plt.savefig('enhanced_training_results.png')
    plt.show()
    
    return agent, best_coverage


if __name__ == "__main__":
    agent, best_coverage = train_enhanced_agent(episodes=10000)
    print(f"\nFinal best coverage: {best_coverage:.1%}")