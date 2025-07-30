import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Append the utils directory to the system path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.algorithms.rl_agent_parallel import CirclePlacementEnv, compute_included

def visualize_reward_function():
    """Visualize how rewards differ for various placement qualities."""
    
    # Create a test map with different value regions
    map_size = 64
    test_map = np.zeros((map_size, map_size))
    
    # Create regions with different values
    # High value region (max density)
    test_map[10:20, 10:20] = 100
    
    # Medium-high value region (75% density)
    test_map[30:40, 10:20] = 75
    
    # Medium value region (50% density)
    test_map[10:20, 30:40] = 50
    
    # Low value region (25% density)
    test_map[30:40, 30:40] = 25
    
    # Very low value region (10% density)
    test_map[45:55, 45:55] = 10
    
    # Test different placements
    radius = 6
    placements = [
        (15, 15, "High (100%)"),
        (35, 15, "Medium-High (75%)"),
        (15, 35, "Medium (50%)"),
        (35, 35, "Low (25%)"),
        (50, 50, "Very Low (10%)"),
        (25, 25, "Edge (partial)"),
        (5, 5, "Empty area"),
    ]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show the test map
    im = ax1.imshow(test_map, cmap='hot')
    ax1.set_title('Test Map with Different Value Regions')
    plt.colorbar(im, ax=ax1)
    
    # Draw circles for each placement
    for x, y, label in placements:
        circle = plt.Circle((y, x), radius, fill=False, color='cyan', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(y, x, label.split()[0], ha='center', va='center', color='cyan', fontsize=8)
    
    # Calculate rewards for each placement
    rewards = []
    densities = []
    
    for x, y, label in placements:
        # Create a copy of the map for this test
        map_copy = test_map.copy()
        
        # Calculate what the reward would be
        included_weight = compute_included(map_copy, x, y, radius)
        
        # Calculate density ratio
        cells_in_circle = 0
        total_original_weight = 0
        for i in range(max(0, int(x - radius)), min(map_size, int(x + radius + 1))):
            for j in range(max(0, int(y - radius)), min(map_size, int(y + radius + 1))):
                if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                    cells_in_circle += 1
                    total_original_weight += test_map[i, j]
        
        avg_weight_density = total_original_weight / max(cells_in_circle, 1)
        max_density = test_map.max()
        density_ratio = avg_weight_density / max(max_density, 1)
        
        # Calculate reward using the new formula
        if included_weight <= 0:
            reward = -1.0
        else:
            quality_bonus = density_ratio ** 2
            base_reward = quality_bonus
            collection_bonus = min(included_weight / (np.pi * radius * radius * max_density), 0.2)
            reward = base_reward + collection_bonus
            
            if density_ratio > 0.8:
                reward += 0.3
        
        rewards.append(reward)
        densities.append(density_ratio)
        
        print(f"\n{label}:")
        print(f"  Position: ({x}, {y})")
        print(f"  Weight collected: {included_weight:.1f}")
        print(f"  Density ratio: {density_ratio:.3f}")
        print(f"  Reward: {reward:.3f}")
    
    # Plot rewards
    labels = [p[2] for p in placements]
    colors = plt.cm.RdYlGn([(r + 1) / 2.5 for r in rewards])  # Normalize to 0-1 for colormap
    
    bars = ax2.bar(range(len(rewards)), rewards, color=colors)
    ax2.set_xticks(range(len(rewards)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('Reward')
    ax2.set_title('Rewards for Different Placement Qualities')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, reward) in enumerate(zip(bars, rewards)):
        height = bar.get_height()
        y_pos = height + 0.02 if height > 0 else height - 0.05
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{reward:.2f}', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('reward_function_visualization.png', dpi=150)
    plt.show()
    
    # Show reward curve
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    
    # Generate density ratios from 0 to 1
    density_ratios = np.linspace(0, 1, 100)
    rewards_curve = []
    
    for dr in density_ratios:
        quality_bonus = dr ** 2
        base_reward = quality_bonus
        collection_bonus = 0.1  # Average collection bonus
        reward = base_reward + collection_bonus
        if dr > 0.8:
            reward += 0.3
        rewards_curve.append(reward)
    
    ax3.plot(density_ratios * 100, rewards_curve, linewidth=2)
    ax3.set_xlabel('Area Density (%)')
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward Function: Non-linear Preference for High-Value Areas')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=80, color='red', linestyle='--', alpha=0.5, label='Bonus threshold (80%)')
    ax3.legend()
    
    # Add annotations
    ax3.annotate('Low value\n(weak reward)', xy=(25, 0.15), xytext=(25, 0.4),
                arrowprops=dict(arrowstyle='->', color='red'))
    ax3.annotate('High value\n(strong reward)', xy=(90, 1.3), xytext=(70, 1.1),
                arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    plt.savefig('reward_curve.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    visualize_reward_function()