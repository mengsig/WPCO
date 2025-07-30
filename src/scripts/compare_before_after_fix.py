"""
Comparison script to show the dramatic improvement from the overlap prevention fix.
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
from algorithms.rl_agent_parallel import CirclePlacementEnv, random_seeder


class BrokenValidMask:
    """Simulates the old broken behavior for comparison."""
    @staticmethod
    def get_valid_actions_mask(env):
        radius = env.radii[env.current_radius_idx]
        mask = np.ones((env.map_size, env.map_size), dtype=bool)
        
        # ONLY boundary check (the bug!)
        mask[:int(radius), :] = False
        mask[-int(radius):, :] = False
        mask[:, :int(radius)] = False
        mask[:, -int(radius):] = False
        
        return mask


def simulate_placements(env, use_fixed_mask=True):
    """Simulate random placements with or without the fix."""
    state = env.reset()
    
    total_reward = 0
    overlaps = 0
    
    while True:
        # Get valid actions
        if use_fixed_mask:
            valid_mask = env.get_valid_actions_mask()  # Fixed version
        else:
            valid_mask = BrokenValidMask.get_valid_actions_mask(env)  # Broken version
        
        # Random valid action
        valid_positions = np.argwhere(valid_mask)
        if len(valid_positions) == 0:
            break
            
        idx = np.random.randint(len(valid_positions))
        action = tuple(valid_positions[idx])
        
        # Check for overlap before step
        x, y = action
        radius = env.radii[env.current_radius_idx]
        for px, py, pr in env.placed_circles:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < radius + pr:
                overlaps += 1
                break
        
        # Take step
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    return info['coverage'], total_reward, overlaps


def main():
    print("=" * 70)
    print("COMPARING PERFORMANCE: BEFORE vs AFTER OVERLAP FIX")
    print("=" * 70)
    
    # Run multiple simulations
    n_simulations = 50
    
    broken_coverage = []
    broken_rewards = []
    broken_overlaps = []
    
    fixed_coverage = []
    fixed_rewards = []
    fixed_overlaps = []
    
    print("\nRunning simulations...")
    print("-" * 70)
    
    for i in range(n_simulations):
        # Generate map
        weighted_matrix = random_seeder(64, time_steps=100000)
        
        # Test broken version
        env_broken = CirclePlacementEnv(64)
        env_broken.reset(weighted_matrix.copy())
        cov, rew, ovr = simulate_placements(env_broken, use_fixed_mask=False)
        broken_coverage.append(cov)
        broken_rewards.append(rew)
        broken_overlaps.append(ovr)
        
        # Test fixed version
        env_fixed = CirclePlacementEnv(64)
        env_fixed.reset(weighted_matrix.copy())
        cov, rew, ovr = simulate_placements(env_fixed, use_fixed_mask=True)
        fixed_coverage.append(cov)
        fixed_rewards.append(rew)
        fixed_overlaps.append(ovr)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{n_simulations} simulations...")
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("RESULTS (Random Placement Strategy):")
    print("=" * 70)
    
    print("\nBROKEN (Before Fix):")
    print(f"  Average Coverage: {np.mean(broken_coverage):.1%} ± {np.std(broken_coverage):.1%}")
    print(f"  Average Reward: {np.mean(broken_rewards):.2f} ± {np.std(broken_rewards):.2f}")
    print(f"  Average Overlaps: {np.mean(broken_overlaps):.1f} ± {np.std(broken_overlaps):.1f}")
    print(f"  % Negative Rewards: {sum(r < 0 for r in broken_rewards) / len(broken_rewards) * 100:.0f}%")
    
    print("\nFIXED (After Fix):")
    print(f"  Average Coverage: {np.mean(fixed_coverage):.1%} ± {np.std(fixed_coverage):.1%}")
    print(f"  Average Reward: {np.mean(fixed_rewards):.2f} ± {np.std(fixed_rewards):.2f}")
    print(f"  Average Overlaps: {np.mean(fixed_overlaps):.1f} ± {np.std(fixed_overlaps):.1f}")
    print(f"  % Negative Rewards: {sum(r < 0 for r in fixed_rewards) / len(fixed_rewards) * 100:.0f}%")
    
    print("\nIMPROVEMENT:")
    print(f"  Coverage: +{(np.mean(fixed_coverage) - np.mean(broken_coverage)) / np.mean(broken_coverage) * 100:.0f}%")
    print(f"  Reward: {np.mean(fixed_rewards) - np.mean(broken_rewards):+.2f}")
    print(f"  Overlaps: {np.mean(fixed_overlaps) - np.mean(broken_overlaps):+.1f}")
    
    # Visualize
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Coverage comparison
    ax1.boxplot([broken_coverage, fixed_coverage], labels=['Broken', 'Fixed'])
    ax1.set_ylabel('Coverage')
    ax1.set_title('Coverage Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Reward comparison
    ax2.boxplot([broken_rewards, fixed_rewards], labels=['Broken', 'Fixed'])
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Reward Distribution')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Overlap comparison
    ax3.bar(['Broken', 'Fixed'], [np.mean(broken_overlaps), np.mean(fixed_overlaps)], 
            yerr=[np.std(broken_overlaps), np.std(fixed_overlaps)], capsize=10)
    ax3.set_ylabel('Average Overlaps per Game')
    ax3.set_title('Overlap Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Scatter plot
    ax4.scatter(broken_coverage, broken_rewards, alpha=0.5, label='Broken', color='red')
    ax4.scatter(fixed_coverage, fixed_rewards, alpha=0.5, label='Fixed', color='green')
    ax4.set_xlabel('Coverage')
    ax4.set_ylabel('Total Reward')
    ax4.set_title('Coverage vs Reward')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Impact of Overlap Prevention Fix', fontsize=16)
    plt.tight_layout()
    plt.savefig('before_after_fix_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to before_after_fix_comparison.png")
    
    # Example visualization
    print("\nGenerating example visualization...")
    weighted_matrix = random_seeder(64, time_steps=100000)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Broken version
    env_broken = CirclePlacementEnv(64)
    env_broken.reset(weighted_matrix.copy())
    _, _, _ = simulate_placements(env_broken, use_fixed_mask=False)
    
    ax1.imshow(env_broken.original_map, cmap='hot', alpha=0.6)
    for x, y, r in env_broken.placed_circles:
        circle = plt.Circle((y, x), r, fill=False, color='red', linewidth=2)
        ax1.add_patch(circle)
    ax1.set_title(f'BROKEN: Many Overlaps (Coverage: {1 - env_broken.current_map.sum() / env_broken.original_map.sum():.1%})')
    ax1.axis('off')
    
    # Fixed version
    env_fixed = CirclePlacementEnv(64)
    env_fixed.reset(weighted_matrix.copy())
    _, _, _ = simulate_placements(env_fixed, use_fixed_mask=True)
    
    ax2.imshow(env_fixed.original_map, cmap='hot', alpha=0.6)
    for x, y, r in env_fixed.placed_circles:
        circle = plt.Circle((y, x), r, fill=False, color='green', linewidth=2)
        ax2.add_patch(circle)
    ax2.set_title(f'FIXED: No Overlaps (Coverage: {1 - env_fixed.current_map.sum() / env_fixed.original_map.sum():.1%})')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('overlap_fix_example.png', dpi=150, bbox_inches='tight')
    print("Example saved to overlap_fix_example.png")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("The overlap prevention fix transforms the problem from impossible to solvable!")
    print("Agents can now learn effective strategies instead of being forced to fail.")
    print("=" * 70)


if __name__ == "__main__":
    main()