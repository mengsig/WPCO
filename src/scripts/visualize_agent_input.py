import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

# Append the utils directory to the system path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.algorithms.rl_agent_parallel import ImprovedDQNAgent, CirclePlacementEnv, random_seeder

def visualize_agent_input():
    """Visualize exactly what the agent sees as input."""
    
    # Create environment
    env = CirclePlacementEnv(map_size=32, radii=[8, 6, 4, 3, 2])
    
    # Generate a simple test map
    weighted_matrix = np.zeros((32, 32))
    # Create some high-value regions
    weighted_matrix[8:16, 8:16] = 10    # Center square
    weighted_matrix[20:25, 5:10] = 8    # Top-left region
    weighted_matrix[5:10, 20:28] = 6    # Bottom-right region
    
    # Reset environment
    state = env.reset(weighted_matrix)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # Simulate placing circles
    placements = [(12, 12), (22, 7), (7, 24), (20, 20), (25, 25)]
    
    for step in range(len(env.radii)):
        # Get current state
        state = env._get_state()
        
        # Extract components
        state_map = state[:-2].reshape(32, 32)
        current_radius_normalized = state[-2]
        num_placed_normalized = state[-1]
        
        # Current radius (actual value)
        current_radius = env.radii[env.current_radius_idx]
        
        # Create subplot for this step
        ax_idx = step * 3
        
        # 1. Original map (what exists in memory)
        ax1 = plt.subplot(5, 3, ax_idx + 1)
        im1 = ax1.imshow(env.original_map, cmap='hot')
        ax1.set_title(f'Step {step+1}: Original Map')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # 2. Current map (after modifications)
        ax2 = plt.subplot(5, 3, ax_idx + 2)
        im2 = ax2.imshow(env.current_map, cmap='hot')
        ax2.set_title(f'Current Map (zeros where circles placed)')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # 3. What agent sees
        ax3 = plt.subplot(5, 3, ax_idx + 3)
        im3 = ax3.imshow(state_map, cmap='RdBu_r', vmin=-1, vmax=1)
        ax3.set_title(f'Agent Input: r={current_radius} ({current_radius_normalized:.2f}), placed={step}/{len(env.radii)} ({num_placed_normalized:.2f})')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Add circle outline to show what we're about to place
        circle = plt.Circle(placements[step], current_radius, fill=False, color='yellow', linewidth=2)
        ax3.add_patch(circle)
        
        # Print state info
        print(f"\nStep {step + 1}:")
        print(f"  Radius to place: {current_radius}")
        print(f"  Radius (normalized): {current_radius_normalized:.3f}")
        print(f"  Circles placed: {step}/{len(env.radii)}")
        print(f"  Progress (normalized): {num_placed_normalized:.3f}")
        print(f"  State map range: [{state_map.min():.3f}, {state_map.max():.3f}]")
        print(f"  Number of -1 values: {(state_map == -1).sum()}")
        print(f"  Action: {placements[step]}")
        
        # Take action
        next_state, reward, done, info = env.step(placements[step])
        print(f"  Reward: {reward:.3f}")
        print(f"  Coverage so far: {info['coverage_ratio']:.3f}")
        
        if done:
            break
    
    plt.tight_layout()
    plt.savefig('agent_input_visualization.png', dpi=150)
    plt.show()
    
    # Show the full state vector structure
    print("\n" + "="*60)
    print("STATE VECTOR STRUCTURE:")
    print("="*60)
    print(f"Total state size: {len(state)}")
    print(f"  - Map pixels: {32*32} = 1024")
    print(f"  - Current radius (normalized): 1")
    print(f"  - Progress (normalized): 1")
    print(f"  - Total: 1026")
    print("\nThe agent sees:")
    print("  - Map values: -1 (already placed), 0 (empty), 0-1 (available weights)")
    print("  - Current radius: normalized to [0, 1] by dividing by max radius")
    print("  - Progress: normalized to [0, 1] as fraction of circles placed")

if __name__ == "__main__":
    visualize_agent_input()