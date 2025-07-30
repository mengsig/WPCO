import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Append the utils directory to the system path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.algorithms.rl_agent_parallel import CirclePlacementEnv, random_seeder

def debug_masking():
    """Debug the masking to see what the agent actually sees."""
    
    # Create environment
    env = CirclePlacementEnv(map_size=32, radii=[8, 6, 4, 3])  # Smaller for easier visualization
    
    # Generate a simple map with clear high-value area
    weighted_matrix = np.zeros((32, 32))
    weighted_matrix[10:20, 10:20] = 10  # High value square in center
    weighted_matrix[5:10, 20:25] = 8    # Another high value area
    
    # Reset environment
    state = env.reset(weighted_matrix)
    
    # Create figure for visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    print("Initial state shape:", state.shape)
    print("Map values - Original max:", env.original_map.max(), "Current max:", env.current_map.max())
    
    # Step through placing each circle
    for step in range(len(env.radii)):
        # Get current state
        state = env._get_state()
        
        # Extract the map from state (all but last 2 elements)
        state_map = state[:-2].reshape(32, 32)
        
        # Visualize original map
        ax1 = axes[step * 2]
        im1 = ax1.imshow(env.current_map, cmap='hot')
        ax1.set_title(f'Step {step}: Current Map\n(before placing radius {env.radii[step]})')
        plt.colorbar(im1, ax=ax1)
        
        # Visualize what agent sees
        ax2 = axes[step * 2 + 1]
        im2 = ax2.imshow(state_map, cmap='RdBu', vmin=-1, vmax=1)
        ax2.set_title(f'Step {step}: Agent View\n(blue = -1 for placed)')
        plt.colorbar(im2, ax=ax2)
        
        # Print some debug info
        print(f"\nStep {step}:")
        print(f"  Radius to place: {env.radii[step]}")
        print(f"  Current map - min: {env.current_map.min():.2f}, max: {env.current_map.max():.2f}")
        print(f"  State map - min: {state_map.min():.2f}, max: {state_map.max():.2f}")
        print(f"  Number of -1 values in state: {(state_map == -1).sum()}")
        print(f"  Number of 0 values in current_map: {(env.current_map == 0).sum()}")
        
        # Place circle in high value area
        if step == 0:
            action = (15, 15)  # Center of high value area
        elif step == 1:
            action = (7, 22)   # Other high value area
        else:
            action = (15 + step * 5, 15)  # Offset from first
        
        print(f"  Placing at: {action}")
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        if done:
            break
    
    plt.tight_layout()
    plt.savefig('debug_masking.png', dpi=150)
    plt.show()
    
    # Also check if compute_included is actually modifying the map
    print("\n" + "="*50)
    print("Testing compute_included directly:")
    
    # Create a test map
    test_map = np.ones((10, 10)) * 5
    print("Before compute_included:")
    print(test_map)
    
    # Import and call compute_included
    from src.algorithms.rl_agent_parallel import compute_included
    weight = compute_included(test_map, 5, 5, 3)
    
    print(f"\nAfter compute_included (weight collected: {weight}):")
    print(test_map)
    print(f"Number of zeros: {(test_map == 0).sum()}")

if __name__ == "__main__":
    debug_masking()