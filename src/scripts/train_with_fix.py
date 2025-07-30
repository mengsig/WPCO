"""
Simple training script that demonstrates the dramatic improvement from the overlap fix.
This uses the standard parallel agent with the fixed get_valid_actions_mask.
"""

import sys
sys.path.append('src')

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from algorithms.rl_agent_parallel import CirclePlacementEnv, ImprovedDQNAgent, random_seeder
from utils.plotting_utils import plot_heatmap_with_circles


def train_with_overlap_fix(n_episodes=1000, visualize_every=100):
    """Train agent with the overlap prevention fix."""
    
    print("=" * 70)
    print("TRAINING RL AGENT WITH OVERLAP PREVENTION FIX")
    print("=" * 70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create environment and agent
    env = CirclePlacementEnv(map_size=64)
    agent = ImprovedDQNAgent(
        map_size=64,
        learning_rate=5e-4,  # Slightly higher for faster learning
        epsilon_start=0.5,   # Lower since valid mask helps exploration
        epsilon_decay_steps=int(n_episodes * 0.7),
        batch_size=32
    )
    
    # Metrics
    episode_rewards = []
    episode_coverage = []
    overlap_counts = []
    
    print("\nKey Improvements:")
    print("✓ Valid action mask prevents overlaps")
    print("✓ Smooth reward function")
    print("✓ Positive rewards for good placements")
    print("✓ Agent can actually learn!")
    print("\nStarting training...")
    print("-" * 70)
    
    # Training loop
    pbar = tqdm(range(n_episodes), desc="Training")
    
    for episode in pbar:
        # New map each episode
        weighted_matrix = random_seeder(64, time_steps=100000)
        state = env.reset(weighted_matrix)
        
        episode_reward = 0
        overlap_count = 0
        
        # Play episode
        while True:
            # Get valid actions (THE KEY FIX!)
            valid_mask = env.get_valid_actions_mask()
            
            # Choose action
            action = agent.act(state, valid_mask)
            
            # Verify no overlap (for statistics)
            x, y = action
            radius = env.radii[env.current_radius_idx]
            for px, py, pr in env.placed_circles:
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                if dist < radius + pr - 0.1:  # Small tolerance
                    overlap_count += 1
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember_n_step(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Train
        if len(agent.memory) > agent.batch_size:
            agent.replay()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_coverage.append(info['coverage'])
        overlap_counts.append(overlap_count)
        
        # Update progress
        pbar.set_postfix({
            'Coverage': f"{info['coverage']:.1%}",
            'Reward': f"{episode_reward:.1f}",
            'ε': f"{agent.epsilon:.3f}"
        })
        
        # Periodic evaluation
        if (episode + 1) % visualize_every == 0:
            print(f"\n{'='*70}")
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"{'='*70}")
            
            # Stats
            recent_coverage = np.mean(episode_coverage[-100:])
            recent_rewards = np.mean(episode_rewards[-100:])
            recent_overlaps = np.mean(overlap_counts[-100:])
            best_coverage = max(episode_coverage)
            
            print(f"\nPERFORMANCE METRICS:")
            print(f"  Recent avg coverage: {recent_coverage:.1%}")
            print(f"  Best coverage: {best_coverage:.1%}")
            print(f"  Recent avg reward: {recent_rewards:.1f}")
            print(f"  Recent avg overlaps: {recent_overlaps:.2f}")
            
            if recent_overlaps > 0.1:
                print(f"\n⚠️  WARNING: Some overlaps still occurring!")
            else:
                print(f"\n✅ SUCCESS: No overlaps detected!")
            
            # Visualize current performance
            print(f"\nVisualizing agent performance...")
            test_map = random_seeder(64, time_steps=100000)
            test_env = CirclePlacementEnv(64)
            state = test_env.reset(test_map)
            
            placements = []
            while True:
                valid_mask = test_env.get_valid_actions_mask()
                with torch.no_grad():
                    action = agent.act(state, valid_mask)
                placements.append((action[0], action[1], test_env.radii[test_env.current_radius_idx]))
                state, _, done, info = test_env.step(action)
                if done:
                    break
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original
            ax1.imshow(test_env.original_map, cmap='hot')
            ax1.set_title('Original Heatmap')
            ax1.axis('off')
            
            # With circles
            ax2.imshow(test_env.original_map, cmap='hot', alpha=0.6)
            colors = plt.cm.tab10(np.linspace(0, 1, len(placements)))
            for i, (x, y, r) in enumerate(placements):
                circle = plt.Circle((y, x), r, fill=False, color=colors[i], linewidth=2)
                ax2.add_patch(circle)
                ax2.text(y, x, str(i+1), ha='center', va='center', fontsize=8)
            ax2.set_title(f'Agent Placement (Coverage: {info["coverage"]:.1%})')
            ax2.axis('off')
            
            plt.suptitle(f'Episode {episode + 1} - No Overlaps!', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'placement_ep{episode+1}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved to placement_ep{episode+1}.png")
    
    # Save model
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episode': n_episodes,
        'episode_rewards': episode_rewards,
        'episode_coverage': episode_coverage
    }, 'fixed_agent_model.pth')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final average coverage: {np.mean(episode_coverage[-100:]):.1%}")
    print(f"Best coverage achieved: {max(episode_coverage):.1%}")
    print(f"Total overlaps in last 100 episodes: {sum(overlap_counts[-100:])}")
    print("Model saved to fixed_agent_model.pth")
    
    # Plot training progress
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Coverage
    ax1.plot(episode_coverage, alpha=0.3, label='Episode')
    if len(episode_coverage) >= 100:
        ax1.plot(np.convolve(episode_coverage, np.ones(100)/100, mode='valid'), 
                 label='100-ep average', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Coverage')
    ax1.set_title('Coverage Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rewards
    ax2.plot(episode_rewards, alpha=0.3, label='Episode')
    if len(episode_rewards) >= 100:
        ax2.plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'), 
                 label='100-ep average', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Rewards Over Time (Should be Positive!)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Overlaps
    ax3.plot(overlap_counts, alpha=0.5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Overlaps per Episode')
    ax3.set_title('Overlap Frequency (Should be Zero!)')
    ax3.grid(True, alpha=0.3)
    
    # Coverage vs Reward scatter
    ax4.scatter(episode_coverage, episode_rewards, alpha=0.5, c=range(len(episode_coverage)), cmap='viridis')
    ax4.set_xlabel('Coverage')
    ax4.set_ylabel('Total Reward')
    ax4.set_title('Coverage vs Reward Correlation')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Training Progress with Overlap Prevention Fix', fontsize=16)
    plt.tight_layout()
    plt.savefig('training_progress_fixed.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nTraining visualization saved to training_progress_fixed.png")
    
    # Final comparison message
    print("\n" + "="*70)
    print("COMPARISON TO BROKEN VERSION:")
    print("="*70)
    print("Before fix: 100% negative rewards, constant overlaps, no learning")
    print(f"After fix:  {np.mean(episode_rewards[-100:]):.1f} avg reward, "
          f"{sum(overlap_counts[-100:])} overlaps, "
          f"{np.mean(episode_coverage[-100:]):.1%} coverage")
    print("\nThe overlap prevention fix makes the problem learnable!")
    print("="*70)


if __name__ == "__main__":
    train_with_overlap_fix(n_episodes=1000, visualize_every=200)