import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_reward(density_ratio, overlap_ratio):
    """Calculate reward using the smooth function"""
    # Base reward: quality of placement (0 to 1)
    quality_score = density_ratio ** 1.5
    
    # For visualization, assume collection_score proportional to density
    collection_score = density_ratio * 0.8
    
    # Combine quality and collection with weights
    base_reward = (0.7 * quality_score + 0.3 * collection_score)
    
    # Apply smooth overlap penalty using exponential decay
    if overlap_ratio > 0:
        overlap_penalty = 1 - np.exp(-3 * overlap_ratio)
        reward = base_reward * (1 - overlap_penalty) - 0.1 * overlap_penalty
    else:
        reward = base_reward
    
    # Add small exploration bonus for non-zero density areas
    if density_ratio > 0.1 and overlap_ratio == 0:
        reward += 0.05
    
    # Smooth penalty for empty areas
    if density_ratio < 0.01:
        reward = reward * 0.1 - 0.02
    
    # Final clipping
    reward = np.clip(reward, -0.3, 1.1)
    
    return reward

# Create meshgrid for visualization
density_ratios = np.linspace(0, 1, 50)
overlap_ratios = np.linspace(0, 1, 50)
D, O = np.meshgrid(density_ratios, overlap_ratios)

# Calculate rewards for all combinations
rewards = np.zeros_like(D)
for i in range(len(density_ratios)):
    for j in range(len(overlap_ratios)):
        rewards[j, i] = calculate_reward(D[j, i], O[j, i])

# Create figure with multiple views
fig = plt.figure(figsize=(15, 10))

# 3D surface plot
ax1 = fig.add_subplot(221, projection='3d')
surf = ax1.plot_surface(D, O, rewards, cmap='viridis', alpha=0.8)
ax1.set_xlabel('Density Ratio')
ax1.set_ylabel('Overlap Ratio')
ax1.set_zlabel('Reward')
ax1.set_title('Smooth Reward Function (3D View)')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# Contour plot
ax2 = fig.add_subplot(222)
contour = ax2.contourf(D, O, rewards, levels=20, cmap='viridis')
ax2.set_xlabel('Density Ratio')
ax2.set_ylabel('Overlap Ratio')
ax2.set_title('Reward Contours')
fig.colorbar(contour, ax=ax2)

# Cross-sections at different overlap ratios
ax3 = fig.add_subplot(223)
for overlap in [0, 0.1, 0.3, 0.5, 0.7, 1.0]:
    rewards_slice = [calculate_reward(d, overlap) for d in density_ratios]
    ax3.plot(density_ratios, rewards_slice, label=f'Overlap={overlap:.1f}')
ax3.set_xlabel('Density Ratio')
ax3.set_ylabel('Reward')
ax3.set_title('Reward vs Density at Different Overlaps')
ax3.legend()
ax3.grid(True)

# Cross-sections at different density ratios
ax4 = fig.add_subplot(224)
for density in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    rewards_slice = [calculate_reward(density, o) for o in overlap_ratios]
    ax4.plot(overlap_ratios, rewards_slice, label=f'Density={density:.1f}')
ax4.set_xlabel('Overlap Ratio')
ax4.set_ylabel('Reward')
ax4.set_title('Reward vs Overlap at Different Densities')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.savefig('smooth_reward_function.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some key values
print("Reward Function Analysis:")
print(f"Max reward (density=1, overlap=0): {calculate_reward(1.0, 0.0):.3f}")
print(f"Good placement (density=0.8, overlap=0): {calculate_reward(0.8, 0.0):.3f}")
print(f"Slight overlap (density=0.8, overlap=0.1): {calculate_reward(0.8, 0.1):.3f}")
print(f"Medium overlap (density=0.8, overlap=0.3): {calculate_reward(0.8, 0.3):.3f}")
print(f"High overlap (density=0.8, overlap=0.5): {calculate_reward(0.8, 0.5):.3f}")
print(f"Empty area (density=0, overlap=0): {calculate_reward(0.0, 0.0):.3f}")
print(f"Low density (density=0.1, overlap=0): {calculate_reward(0.1, 0.0):.3f}")