import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt

# Append the utils directory to the system path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.plotting_utils import plot_defaults
from src.algorithms.rl_agent import DQNAgent, CirclePlacementEnv, random_seeder


def demo_rl_agent(model_path=None):
    """
    Demo script showing how to use a trained RL agent.
    """
    plot_defaults()
    
    # Configuration
    map_size = 64
    radii = [12, 8, 7, 6, 5, 4, 3, 2, 1]
    
    # Load trained agent
    if model_path is None:
        model_path = "results/rl_agent/final_model.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the agent first using train_rl_agent.py")
        return
    
    print(f"Loading trained RL agent from {model_path}")
    agent = DQNAgent(map_size=map_size, radii=radii)
    checkpoint = torch.load(model_path)
    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.epsilon = 0.0  # No exploration during demo
    
    # Create environment
    env = CirclePlacementEnv(map_size=map_size, radii=radii)
    
    # Generate a random map
    print("\nGenerating random weighted map...")
    weighted_matrix = random_seeder(map_size, time_steps=100000)
    
    # Reset environment with the map
    state = env.reset(weighted_matrix)
    
    print(f"\nStarting circle placement demo...")
    print(f"Total weight on map: {np.sum(weighted_matrix):.2f}")
    print(f"Circles to place: {radii}")
    
    # Create figure for step-by-step visualization
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    step = 0
    done = False
    
    while not done and step < len(radii):
        # Get valid actions
        valid_mask = env.get_valid_actions_mask()
        
        # Agent selects action
        action = agent.act(state, valid_mask)
        x, y = action
        
        print(f"\nStep {step + 1}: Placing circle with radius {radii[step]} at position ({x}, {y})")
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        
        print(f"  Reward: {reward:.2f}")
        print(f"  Coverage so far: {info['coverage_ratio']:.3f}")
        
        # Visualize current state
        ax = axes[step]
        ax.imshow(env.original_map, cmap='hot', interpolation='nearest')
        
        # Draw all placed circles
        from matplotlib.patches import Circle
        for i, (cx, cy, r) in enumerate(env.placed_circles):
            color = 'blue' if i < len(env.placed_circles) - 1 else 'green'
            alpha = 0.3 if i < len(env.placed_circles) - 1 else 0.5
            circle = Circle((cx, cy), radius=r, color=color, alpha=alpha)
            ax.add_patch(circle)
        
        ax.set_title(f'Step {step + 1}: r={radii[step]}')
        ax.set_xlim(0, map_size)
        ax.set_ylim(map_size, 0)
        ax.axis('off')
        
        state = next_state
        step += 1
    
    # Hide unused subplots
    for i in range(step, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'RL Agent Circle Placement Demo\nFinal Coverage: {info["coverage_ratio"]:.3f}', fontsize=16)
    plt.tight_layout()
    
    # Save the visualization
    save_dir = "results/demo"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/rl_agent_step_by_step.png", dpi=300)
    plt.show()
    
    # Create final summary visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original map
    ax1.imshow(env.original_map, cmap='hot', interpolation='nearest')
    ax1.set_title('Original Weighted Map')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Final placement
    ax2.imshow(env.original_map, cmap='hot', interpolation='nearest')
    for cx, cy, r in env.placed_circles:
        circle = Circle((cx, cy), radius=r, color='blue', alpha=0.5)
        ax2.add_patch(circle)
    ax2.set_title(f'Final Placement (Coverage: {info["coverage_ratio"]:.3f})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_xlim(0, map_size)
    ax2.set_ylim(map_size, 0)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/rl_agent_final_result.png", dpi=300)
    plt.show()
    
    print(f"\nDemo complete!")
    print(f"Final statistics:")
    print(f"  Total weight collected: {info['weight_collected']:.2f}")
    print(f"  Total weight on map: {info['total_weight']:.2f}")
    print(f"  Coverage ratio: {info['coverage_ratio']:.3f}")


if __name__ == "__main__":
    demo_rl_agent()