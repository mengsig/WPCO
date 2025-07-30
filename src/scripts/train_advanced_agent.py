import sys
sys.path.append('src')

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from algorithms.rl_agent_advanced import (
    AdvancedCirclePlacementEnv, GuidedDQNAgent, random_seeder
)
from utils.plotting_utils import plot_heatmap_with_circles
import time


def visualize_strategy(env, agent, episode, save_path='advanced_strategy.png'):
    """Visualize the agent's placement strategy."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original heatmap
    axes[0, 0].imshow(env.original_map, cmap='hot')
    axes[0, 0].set_title('Original Heatmap')
    axes[0, 0].axis('off')
    
    # Show placements step by step
    temp_env = AdvancedCirclePlacementEnv(env.map_size)
    temp_env.reset(env.original_map.copy())
    
    steps_to_show = [2, 5, 8]  # After 3, 6, and 9 circles
    step_idx = 0
    
    for i in range(len(env.radii)):
        state = temp_env._get_enhanced_state()
        valid_mask = temp_env.get_valid_actions_mask()
        
        # Get agent's action
        with torch.no_grad():
            action = agent.act(state, temp_env, valid_mask)
        
        # Take step
        state, reward, done, info = temp_env.step(action)
        
        # Visualize at specific steps
        if i + 1 in steps_to_show:
            row = step_idx // 3
            col = step_idx % 3 + 1
            
            axes[row, col].imshow(temp_env.original_map, cmap='hot', alpha=0.6)
            
            # Draw circles
            for x, y, r in temp_env.placed_circles:
                circle = plt.Circle((y, x), r, fill=False, color='blue', linewidth=2)
                axes[row, col].add_patch(circle)
            
            axes[row, col].set_title(f'After {i+1} circles (Coverage: {info["coverage"]:.1%})')
            axes[row, col].axis('off')
            
            step_idx += 1
    
    # Final result
    axes[1, 0].imshow(temp_env.original_map, cmap='hot', alpha=0.6)
    for x, y, r in temp_env.placed_circles:
        circle = plt.Circle((y, x), r, fill=False, color='blue', linewidth=2)
        axes[1, 0].add_patch(circle)
    axes[1, 0].set_title(f'Final (Coverage: {info["coverage"]:.1%})')
    axes[1, 0].axis('off')
    
    # Feature visualization
    features = state['raw_features']
    axes[1, 1].imshow(temp_env.current_map, cmap='hot')
    axes[1, 1].set_title('Remaining Values')
    axes[1, 1].axis('off')
    
    # Cluster visualization
    if features['clusters']:
        cluster_map = np.zeros_like(temp_env.current_map)
        for i, cluster in enumerate(features['clusters'][:5]):
            for pos in cluster['positions']:
                cluster_map[pos[0], pos[1]] = (i + 1) * 50
        axes[1, 2].imshow(cluster_map, cmap='tab10')
        axes[1, 2].set_title('Value Clusters')
    else:
        axes[1, 2].text(0.5, 0.5, 'No clusters', ha='center', va='center')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Advanced Agent Strategy - Episode {episode}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_advanced_agent(n_episodes=2000, visualize_every=200):
    """Train the advanced agent with human-like heuristics."""
    
    print("=" * 60)
    print("TRAINING ADVANCED RL AGENT WITH HUMAN-LIKE HEURISTICS")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create environment and agent
    env = AdvancedCirclePlacementEnv(map_size=64)
    agent = GuidedDQNAgent(
        map_size=64,
        learning_rate=1e-4,
        epsilon_start=0.5,  # Lower initial epsilon due to guided exploration
        epsilon_decay_steps=int(n_episodes * 0.6),
        batch_size=32,
        use_suggestions=True,
        suggestion_prob=0.3  # 30% of random actions use heuristics
    )
    
    # Training metrics
    episode_rewards = []
    episode_coverage = []
    losses = []
    
    print("\nKey Features:")
    print("- Cluster detection and targeting")
    print("- Value density analysis")
    print("- Strategic feature extraction")
    print("- Guided exploration with human-like heuristics")
    print("- Enhanced reward shaping")
    print("\nStarting training...")
    print("-" * 60)
    
    # Training loop
    pbar = tqdm(range(n_episodes), desc="Training")
    
    for episode in pbar:
        # Generate new map
        weighted_matrix = random_seeder(64, time_steps=100000)
        state = env.reset(weighted_matrix)
        
        episode_reward = 0
        steps = 0
        
        # Play one episode
        while True:
            # Get valid actions
            valid_mask = env.get_valid_actions_mask()
            
            # Choose action
            action = agent.act(state, env, valid_mask)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state if not done else None, done)
            
            # Update
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # Train
        if len(agent.memory) > agent.batch_size:
            loss = agent.replay()
            if loss is not None:
                losses.append(loss)
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_coverage.append(info['coverage'])
        
        # Update progress bar
        pbar.set_postfix({
            'Coverage': f"{info['coverage']:.1%}",
            'Reward': f"{episode_reward:.2f}",
            'Epsilon': f"{agent.epsilon:.3f}"
        })
        
        # Periodic evaluation and visualization
        if (episode + 1) % visualize_every == 0:
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"{'='*60}")
            
            # Performance stats
            recent_coverage = np.mean(episode_coverage[-100:])
            recent_rewards = np.mean(episode_rewards[-100:])
            best_coverage = max(episode_coverage)
            
            print(f"\nPERFORMANCE:")
            print(f"  Recent avg coverage: {recent_coverage:.1%}")
            print(f"  Best coverage: {best_coverage:.1%}")
            print(f"  Recent avg reward: {recent_rewards:.2f}")
            
            if losses:
                print(f"\nTRAINING:")
                print(f"  Recent avg loss: {np.mean(losses[-100:]):.4f}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
            
            # Visualize strategy
            visualize_strategy(env, agent, episode + 1, 
                             f'advanced_strategy_ep{episode+1}.png')
            print(f"\nStrategy visualization saved!")
            
            # Test on a few maps
            print(f"\nTESTING ON 5 RANDOM MAPS:")
            test_coverages = []
            for i in range(5):
                test_map = random_seeder(64, time_steps=100000)
                test_env = AdvancedCirclePlacementEnv(64)
                state = test_env.reset(test_map)
                
                while True:
                    valid_mask = test_env.get_valid_actions_mask()
                    with torch.no_grad():
                        action = agent.act(state, test_env, valid_mask)
                    state, _, done, info = test_env.step(action)
                    if done:
                        test_coverages.append(info['coverage'])
                        break
            
            print(f"  Test coverages: {[f'{c:.1%}' for c in test_coverages]}")
            print(f"  Average: {np.mean(test_coverages):.1%}")
    
    # Save model
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episode': n_episodes,
        'episode_rewards': episode_rewards,
        'episode_coverage': episode_coverage
    }, 'advanced_rl_model.pth')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final average coverage: {np.mean(episode_coverage[-100:]):.1%}")
    print(f"Best coverage achieved: {max(episode_coverage):.1%}")
    print("Model saved to advanced_rl_model.pth")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Coverage over time
    ax1.plot(episode_coverage, alpha=0.3, label='Episode')
    ax1.plot(np.convolve(episode_coverage, np.ones(100)/100, mode='valid'), 
             label='100-episode average', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Coverage')
    ax1.set_title('Coverage Over Training')
    ax1.legend()
    ax1.grid(True)
    
    # Rewards over time
    ax2.plot(episode_rewards, alpha=0.3, label='Episode')
    ax2.plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'), 
             label='100-episode average', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Rewards Over Training')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('advanced_training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    train_advanced_agent(n_episodes=2000, visualize_every=200)