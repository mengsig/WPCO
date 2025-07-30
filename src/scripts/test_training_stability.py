import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Append the utils directory to the system path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.algorithms.rl_agent_parallel import ImprovedDQNAgent, CirclePlacementEnv, random_seeder

def test_training_stability(n_episodes=500):
    """Test training stability with improved hyperparameters."""
    
    print("Testing training stability with improved hyperparameters...")
    print("-" * 80)
    
    # Configuration
    map_size = 64
    radii = [12, 8, 7, 6, 5, 4, 3, 2, 1]
    
    # Create agent with stable hyperparameters
    agent = ImprovedDQNAgent(
        map_size=map_size,
        radii=radii,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=n_episodes // 2,
        buffer_size=10000,  # Smaller for testing
        batch_size=32,
        tau=0.001,
        n_step=1,
        use_double_dqn=True
    )
    
    # Create environment
    env = CirclePlacementEnv(map_size=map_size, radii=radii)
    
    # Metrics
    episode_rewards = []
    episode_coverage = []
    losses = []
    q_values = []
    
    print(f"Training for {n_episodes} episodes to test stability...")
    
    for episode in tqdm(range(n_episodes)):
        # Generate new map
        weighted_matrix = random_seeder(map_size, time_steps=10000)
        state = env.reset(weighted_matrix)
        
        episode_reward = 0
        done = False
        
        while not done:
            # Get valid actions
            valid_mask = env.get_valid_actions_mask()
            
            # Choose action
            action = agent.act(state, valid_mask)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember_n_step(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            
            # Train if enough experiences
            if len(agent.memory) > agent.batch_size and episode % 4 == 0:
                agent.replay()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_coverage.append(info['coverage_ratio'])
        
        if agent.losses:
            losses.append(agent.losses[-1])
        if agent.q_values:
            q_values.append(agent.q_values[-1])
        
        # Print progress
        if (episode + 1) % 50 == 0:
            stats = agent.get_statistics()
            print(f"\nEpisode {episode + 1}:")
            print(f"  Coverage: {info['coverage_ratio']:.4f}")
            print(f"  Avg Coverage (last 50): {np.mean(episode_coverage[-50:]):.4f}")
            print(f"  Reward: {episode_reward:.4f}")
            print(f"  Loss: {stats['avg_loss']:.4f}")
            print(f"  Avg Q-Value: {stats['avg_q_value']:.4f}")
            print(f"  Epsilon: {stats['epsilon']:.4f}")
            print(f"  Learning Rate: {stats['learning_rate']:.6f}")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Coverage
    ax1.plot(episode_coverage, alpha=0.5)
    if len(episode_coverage) > 50:
        moving_avg = np.convolve(episode_coverage, np.ones(50)/50, mode='valid')
        ax1.plot(range(49, len(episode_coverage)), moving_avg, 'r-', linewidth=2)
    ax1.set_title('Coverage Ratio Over Episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Coverage')
    ax1.grid(True, alpha=0.3)
    
    # Rewards
    ax2.plot(episode_rewards, alpha=0.5)
    if len(episode_rewards) > 50:
        moving_avg = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
        ax2.plot(range(49, len(episode_rewards)), moving_avg, 'r-', linewidth=2)
    ax2.set_title('Rewards Over Episodes (Normalized)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.grid(True, alpha=0.3)
    
    # Loss
    if losses:
        ax3.semilogy(losses, alpha=0.5)
        ax3.set_title('Training Loss (Log Scale)')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.grid(True, alpha=0.3)
    
    # Q-Values
    if q_values:
        ax4.plot(q_values, alpha=0.5)
        ax4.set_title('Max Q-Values')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Q-Value')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_stability_test.png', dpi=150)
    plt.show()
    
    # Analyze stability
    print("\n" + "="*80)
    print("STABILITY ANALYSIS")
    print("="*80)
    
    # Check if loss is stable
    if losses:
        recent_losses = losses[-100:] if len(losses) > 100 else losses
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        print(f"\nLoss Statistics (last 100 steps):")
        print(f"  Mean: {loss_mean:.4f}")
        print(f"  Std: {loss_std:.4f}")
        print(f"  CV: {loss_std/loss_mean:.4f}")
        
        if loss_std/loss_mean < 0.5:
            print("  ✓ Loss is stable")
        else:
            print("  ✗ Loss is unstable")
    
    # Check if Q-values are exploding
    if q_values:
        recent_q = q_values[-100:] if len(q_values) > 100 else q_values
        q_max = np.max(recent_q)
        q_mean = np.mean(recent_q)
        print(f"\nQ-Value Statistics (last 100 steps):")
        print(f"  Mean: {q_mean:.4f}")
        print(f"  Max: {q_max:.4f}")
        
        if q_max < 100:  # Reasonable threshold for normalized rewards
            print("  ✓ Q-values are stable")
        else:
            print("  ✗ Q-values may be exploding")
    
    # Check coverage improvement
    early_coverage = np.mean(episode_coverage[:50]) if len(episode_coverage) > 50 else np.mean(episode_coverage)
    late_coverage = np.mean(episode_coverage[-50:]) if len(episode_coverage) > 50 else np.mean(episode_coverage)
    improvement = (late_coverage - early_coverage) / early_coverage * 100
    
    print(f"\nCoverage Improvement:")
    print(f"  Early episodes: {early_coverage:.4f}")
    print(f"  Recent episodes: {late_coverage:.4f}")
    print(f"  Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print("  ✓ Agent is learning")
    else:
        print("  ✗ Agent is not improving")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    test_training_stability(n_episodes=500)