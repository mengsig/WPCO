"""
Test script to verify that the overlap prevention fix is working correctly.
This will train a simple agent and show that it no longer places overlapping circles.
"""

import sys
sys.path.append('src')

import numpy as np
import torch
from algorithms.rl_agent_parallel import CirclePlacementEnv, ImprovedDQNAgent, random_seeder
import matplotlib.pyplot as plt


def test_overlap_fix(n_episodes=100):
    """Quick test to verify agents no longer overlap circles."""
    
    print("=" * 60)
    print("TESTING OVERLAP PREVENTION FIX")
    print("=" * 60)
    
    # Create environment and agent
    env = CirclePlacementEnv(map_size=64)
    agent = ImprovedDQNAgent(
        map_size=64,
        learning_rate=1e-3,  # Higher for faster learning
        epsilon_start=0.3,   # Lower since we have valid mask
        epsilon_decay_steps=50,
        batch_size=32
    )
    
    # Track metrics
    episode_rewards = []
    episode_coverage = []
    overlaps_detected = []
    
    print("\nRunning quick training test...")
    print("With the fix, we should see:")
    print("- NO overlap warnings")
    print("- Positive rewards")
    print("- Improving coverage")
    print("-" * 60)
    
    for episode in range(n_episodes):
        # New map
        weighted_matrix = random_seeder(64, time_steps=100000)
        state = env.reset(weighted_matrix)
        
        episode_reward = 0
        has_overlap = False
        
        # Play episode
        while True:
            # Get valid actions (THIS IS THE KEY FIX!)
            valid_mask = env.get_valid_actions_mask()
            
            # Choose action
            action = agent.act(state, valid_mask)
            
            # Check if this action would cause overlap
            x, y = action
            radius = env.radii[env.current_radius_idx]
            for px, py, pr in env.placed_circles:
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                if dist < radius + pr:
                    has_overlap = True
                    print(f"WARNING: Overlap detected at episode {episode}!")
                    break
            
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
        overlaps_detected.append(has_overlap)
        
        # Progress update
        if (episode + 1) % 20 == 0:
            recent_rewards = np.mean(episode_rewards[-20:])
            recent_coverage = np.mean(episode_coverage[-20:])
            recent_overlaps = sum(overlaps_detected[-20:])
            
            print(f"\nEpisode {episode + 1}:")
            print(f"  Avg Reward: {recent_rewards:.2f}")
            print(f"  Avg Coverage: {recent_coverage:.1%}")
            print(f"  Overlaps in last 20: {recent_overlaps}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    print(f"Total overlaps detected: {sum(overlaps_detected)}")
    print(f"Final avg coverage: {np.mean(episode_coverage[-20:]):.1%}")
    print(f"Final avg reward: {np.mean(episode_rewards[-20:]):.2f}")
    
    # Visualize one final game
    print("\nVisualizing final placement...")
    state = env.reset(weighted_matrix)
    placements = []
    
    while True:
        valid_mask = env.get_valid_actions_mask()
        with torch.no_grad():
            action = agent.act(state, valid_mask)
        placements.append((action[0], action[1], env.radii[env.current_radius_idx]))
        state, _, done, info = env.step(action)
        if done:
            break
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original map
    ax1.imshow(env.original_map, cmap='hot')
    ax1.set_title('Original Heatmap')
    ax1.axis('off')
    
    # With circles
    ax2.imshow(env.original_map, cmap='hot', alpha=0.6)
    for x, y, r in placements:
        circle = plt.Circle((y, x), r, fill=False, color='blue', linewidth=2)
        ax2.add_patch(circle)
    ax2.set_title(f'Agent Placement (Coverage: {info["coverage"]:.1%})')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('overlap_fix_test.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to overlap_fix_test.png")
    
    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(episode_rewards)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Rewards Over Time (should be mostly positive)')
    ax1.grid(True)
    
    ax2.plot(episode_coverage)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Coverage')
    ax2.set_title('Coverage Over Time (should improve)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('overlap_fix_metrics.png', dpi=150, bbox_inches='tight')
    
    if sum(overlaps_detected) == 0:
        print("\n✅ SUCCESS: No overlaps detected!")
        print("The fix is working correctly.")
    else:
        print("\n❌ FAILURE: Overlaps still occurring!")
        print("The fix needs more work.")


if __name__ == "__main__":
    test_overlap_fix(n_episodes=100)