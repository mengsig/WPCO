"""
Analyze the coverage potential with the correct random_seeder implementation.
Compare what's achievable with different strategies.
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from algorithms.rl_agent_parallel import CirclePlacementEnv, compute_included


@njit()
def random_seeder(dim, time_steps=10000):
    """The correct random_seeder that creates good clustered maps."""
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


def greedy_placement(env):
    """Simple greedy strategy: place each circle at the position with maximum value."""
    total_coverage = []
    
    while True:
        radius = env.radii[env.current_radius_idx]
        best_value = -1
        best_pos = None
        
        # Find position with maximum potential value
        for x in range(radius, env.map_size - radius):
            for y in range(radius, env.map_size - radius):
                # Check if valid (no overlap)
                valid = True
                for px, py, pr in env.placed_circles:
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    if dist < radius + pr:
                        valid = False
                        break
                
                if valid:
                    # Calculate potential value
                    potential = 0
                    for i in range(max(0, x-radius), min(env.map_size, x+radius+1)):
                        for j in range(max(0, y-radius), min(env.map_size, y+radius+1)):
                            if (i-x)**2 + (j-y)**2 <= radius**2:
                                potential += env.current_map[i, j]
                    
                    if potential > best_value:
                        best_value = potential
                        best_pos = (x, y)
        
        if best_pos is None:
            break
            
        # Place circle
        state, reward, done, info = env.step(best_pos)
        total_coverage.append(info['coverage'])
        
        if done:
            break
    
    return info['coverage'], total_coverage


def analyze_maps(n_maps=20):
    """Analyze coverage potential on multiple maps."""
    print("=" * 70)
    print("ANALYZING COVERAGE POTENTIAL WITH CORRECT MAPS")
    print("=" * 70)
    
    greedy_coverages = []
    map_complexities = []
    
    for i in range(n_maps):
        # Generate map with correct seeder
        weighted_matrix = random_seeder(64, time_steps=100000)
        
        # Analyze map complexity
        total_value = weighted_matrix.sum()
        max_value = weighted_matrix.max()
        sparsity = np.sum(weighted_matrix > 0.1) / (64 * 64)
        
        map_complexities.append({
            'total': total_value,
            'max': max_value,
            'sparsity': sparsity
        })
        
        # Test greedy placement
        env = CirclePlacementEnv(64)
        env.reset(weighted_matrix.copy())
        coverage, _ = greedy_placement(env)
        greedy_coverages.append(coverage)
        
        if (i + 1) % 5 == 0:
            print(f"Analyzed {i + 1}/{n_maps} maps...")
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    
    print(f"\nGREEDY PLACEMENT COVERAGE:")
    print(f"  Average: {np.mean(greedy_coverages):.1%}")
    print(f"  Std Dev: {np.std(greedy_coverages):.1%}")
    print(f"  Min: {np.min(greedy_coverages):.1%}")
    print(f"  Max: {np.max(greedy_coverages):.1%}")
    
    print(f"\nMAP CHARACTERISTICS:")
    print(f"  Avg Total Value: {np.mean([m['total'] for m in map_complexities]):.0f}")
    print(f"  Avg Max Value: {np.mean([m['max'] for m in map_complexities]):.1f}")
    print(f"  Avg Sparsity: {np.mean([m['sparsity'] for m in map_complexities]):.1%}")
    
    # Visualize example maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        
        # Generate map
        weighted_matrix = random_seeder(64, time_steps=100000)
        env = CirclePlacementEnv(64)
        env.reset(weighted_matrix.copy())
        coverage, coverage_history = greedy_placement(env)
        
        # Plot
        ax.imshow(env.original_map, cmap='hot', alpha=0.6)
        
        # Draw circles with different colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(env.placed_circles)))
        for i, (x, y, r) in enumerate(env.placed_circles):
            circle = plt.Circle((y, x), r, fill=False, color=colors[i], linewidth=2)
            ax.add_patch(circle)
        
        ax.set_title(f'Coverage: {coverage:.1%}')
        ax.axis('off')
    
    plt.suptitle('Greedy Placement on Correct Maps', fontsize=16)
    plt.tight_layout()
    plt.savefig('coverage_potential_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to coverage_potential_analysis.png")
    
    # Plot coverage progression
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Run one detailed example
    weighted_matrix = random_seeder(64, time_steps=100000)
    env = CirclePlacementEnv(64)
    env.reset(weighted_matrix.copy())
    coverage, coverage_history = greedy_placement(env)
    
    ax.plot(range(1, len(coverage_history) + 1), coverage_history, 'o-', linewidth=2)
    ax.set_xlabel('Circle Number')
    ax.set_ylabel('Coverage')
    ax.set_title('Coverage Progression with Greedy Placement')
    ax.grid(True, alpha=0.3)
    
    # Add circle sizes as annotations
    for i, (cov, radius) in enumerate(zip(coverage_history, env.radii)):
        ax.annotate(f'r={radius}', (i+1, cov), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('coverage_progression.png', dpi=150, bbox_inches='tight')
    print("Coverage progression saved to coverage_progression.png")
    
    print("\n" + "=" * 70)
    print("IMPLICATIONS:")
    print("=" * 70)
    print(f"1. With correct maps, greedy achieves ~{np.mean(greedy_coverages):.0%} coverage")
    print(f"2. Your RL agent achieving ~29% suggests room for improvement")
    print(f"3. The maps have strong clusters that should be targeted")
    print(f"4. Larger circles (r=12, 8, 7) are crucial for good coverage")
    print("=" * 70)


if __name__ == "__main__":
    analyze_maps(n_maps=20)