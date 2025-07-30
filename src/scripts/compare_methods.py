import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt
from time import time

# Append the utils directory to the system path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.plotting_utils import (
    plot_defaults,
    plot_heatmap_with_circles,
)
from src.losses.loss_functions import weighted_loss_function
from src.algorithms.bho import BeehiveOptimization
from src.algorithms.pso import PSO
from src.algorithms.rl_agent import DQNAgent, CirclePlacementEnv, random_seeder


def evaluate_rl_on_map(agent, weighted_matrix, radii):
    """
    Evaluate the RL agent on a specific map.
    """
    env = CirclePlacementEnv(map_size=weighted_matrix.shape[0], radii=radii)
    
    # Set to evaluation mode (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    state = env.reset(weighted_matrix.copy())
    done = False
    
    while not done:
        valid_mask = env.get_valid_actions_mask()
        action = agent.act(state, valid_mask)
        next_state, reward, done, info = env.step(action)
        state = next_state
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    # Convert placements to position array
    pos = []
    for x, y, r in env.placed_circles:
        pos.extend([x, y])
    pos = np.array(pos)
    
    # Calculate loss using the same function as optimization methods
    loss = weighted_loss_function(pos, radii, weighted_matrix.copy())
    
    return pos, loss, info['coverage_ratio']


def compare_methods(n_maps=5, map_size=64, radii=None, rl_model_path=None):
    """
    Compare RL agent with BHO and PSO on the same maps.
    """
    if radii is None:
        radii = [12, 8, 7, 6, 5, 4, 3, 2, 1]
    
    plot_defaults()
    
    # Load trained RL agent
    if rl_model_path is None:
        rl_model_path = "results/rl_agent/final_model.pt"
    
    if not os.path.exists(rl_model_path):
        print(f"RL model not found at {rl_model_path}. Please train the RL agent first.")
        return
    
    agent = DQNAgent(map_size=map_size, radii=radii)
    checkpoint = torch.load(rl_model_path)
    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.epsilon = 0.0  # No exploration during evaluation
    print(f"Loaded RL agent from {rl_model_path}")
    
    # Parameters for optimization methods
    n_particles = 50
    n_iterations = 1000
    dimension = 2
    num_circles = len(radii)
    dim = int(dimension * num_circles)
    
    # Results storage
    results = {
        'rl': {'losses': [], 'coverages': [], 'times': []},
        'bho': {'losses': [], 'coverages': [], 'times': []},
        'pso': {'losses': [], 'coverages': [], 'times': []}
    }
    
    print(f"\nComparing methods on {n_maps} randomly generated maps...")
    
    for map_idx in range(n_maps):
        print(f"\n--- Map {map_idx + 1}/{n_maps} ---")
        
        # Generate random map
        weighted_matrix = random_seeder(map_size, time_steps=100000)
        total_weight = np.sum(weighted_matrix)
        
        # Set bounds for optimization methods
        low_val = 0
        lower = np.ones(dim) * low_val
        upper = np.ones(dim) * low_val
        high_val_x, high_val_y = weighted_matrix.shape[0], weighted_matrix.shape[1]
        for i in range(dim):
            if i % 2 == 0:
                upper[i] = high_val_x
            else:
                upper[i] = high_val_y
        
        # Evaluate RL agent
        print("  Running RL agent...")
        start_time = time()
        rl_pos, rl_loss, rl_coverage = evaluate_rl_on_map(agent, weighted_matrix, radii)
        rl_time = time() - start_time
        results['rl']['losses'].append(rl_loss)
        results['rl']['coverages'].append(rl_coverage)
        results['rl']['times'].append(rl_time)
        print(f"    Loss: {rl_loss:.4f}, Coverage: {rl_coverage:.4f}, Time: {rl_time:.2f}s")
        
        # Run BHO
        print("  Running BHO...")
        start_time = time()
        bho = BeehiveOptimization(
            loss_func=weighted_loss_function,
            weighted_matrix=weighted_matrix.copy(),
            radii=radii,
            n_particles=n_particles,
            dim=dim,
            n_iterations=n_iterations,
            rho=0.99,
            c=0.5,
            q=0.1,
            gamma=0.5,
            dt=0.25,
            init_range=10.0,
            lower_bounds=lower,
            upper_bounds=upper,
            initial_guess=[],
            seed=42
        )
        bho_pos, bho_loss = bho.run()
        bho_time = time() - start_time
        
        # Calculate BHO coverage
        weighted_collected = 0
        temp_matrix = weighted_matrix.copy()
        pos_reshape = np.reshape(bho_pos, (len(radii), 2))
        for idx, radius in enumerate(radii):
            from src.algorithms.rl_agent import compute_included
            cx, cy = pos_reshape[idx]
            weighted_collected += compute_included(temp_matrix, cx, cy, radius)
        bho_coverage = weighted_collected / total_weight if total_weight > 0 else 0
        
        results['bho']['losses'].append(bho_loss)
        results['bho']['coverages'].append(bho_coverage)
        results['bho']['times'].append(bho_time)
        print(f"    Loss: {bho_loss:.4f}, Coverage: {bho_coverage:.4f}, Time: {bho_time:.2f}s")
        
        # Run PSO
        print("  Running PSO...")
        start_time = time()
        pso = PSO(
            loss_func=weighted_loss_function,
            n_particles=n_particles,
            weighted_matrix=weighted_matrix.copy(),
            radii=radii,
            dim=dim,
            n_iterations=n_iterations,
            w=0.7,
            c1=1.5,
            c2=1.5,
            init_range=10.0,
            lower_bounds=lower,
            upper_bounds=upper,
            initial_guess=[],
            seed=42
        )
        pso_pos, pso_loss = pso.run()
        pso_time = time() - start_time
        
        # Calculate PSO coverage
        weighted_collected = 0
        temp_matrix = weighted_matrix.copy()
        pos_reshape = np.reshape(pso_pos, (len(radii), 2))
        for idx, radius in enumerate(radii):
            cx, cy = pos_reshape[idx]
            weighted_collected += compute_included(temp_matrix, cx, cy, radius)
        pso_coverage = weighted_collected / total_weight if total_weight > 0 else 0
        
        results['pso']['losses'].append(pso_loss)
        results['pso']['coverages'].append(pso_coverage)
        results['pso']['times'].append(pso_time)
        print(f"    Loss: {pso_loss:.4f}, Coverage: {pso_coverage:.4f}, Time: {pso_time:.2f}s")
        
        # Save visualizations for this map
        if map_idx < 3:  # Only save first 3 maps to avoid too many files
            save_dir = f"results/comparison/map_{map_idx + 1}"
            os.makedirs(save_dir, exist_ok=True)
            
            plot_heatmap_with_circles(
                weighted_matrix, rl_pos, radii, 
                savename=f"{save_dir}/rl_solution.png"
            )
            plot_heatmap_with_circles(
                weighted_matrix, bho_pos, radii, 
                savename=f"{save_dir}/bho_solution.png"
            )
            plot_heatmap_with_circles(
                weighted_matrix, pso_pos, radii, 
                savename=f"{save_dir}/pso_solution.png"
            )
    
    # Print summary statistics
    print("\n=== SUMMARY RESULTS ===")
    for method in ['rl', 'bho', 'pso']:
        print(f"\n{method.upper()}:")
        print(f"  Average Loss: {np.mean(results[method]['losses']):.4f} ± {np.std(results[method]['losses']):.4f}")
        print(f"  Average Coverage: {np.mean(results[method]['coverages']):.4f} ± {np.std(results[method]['coverages']):.4f}")
        print(f"  Average Time: {np.mean(results[method]['times']):.2f}s ± {np.std(results[method]['times']):.2f}s")
    
    # Create comparison plots
    save_dir = "results/comparison"
    os.makedirs(save_dir, exist_ok=True)
    
    # Coverage comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['RL', 'BHO', 'PSO']
    coverages = [results['rl']['coverages'], results['bho']['coverages'], results['pso']['coverages']]
    times = [results['rl']['times'], results['bho']['times'], results['pso']['times']]
    
    # Box plot of coverages
    ax1.boxplot(coverages, labels=methods)
    ax1.set_ylabel('Coverage Ratio')
    ax1.set_title('Coverage Comparison Across Methods')
    ax1.grid(True, alpha=0.3)
    
    # Bar plot of average times
    avg_times = [np.mean(t) for t in times]
    bars = ax2.bar(methods, avg_times)
    ax2.set_ylabel('Average Time (seconds)')
    ax2.set_title('Computation Time Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/method_comparison.png", dpi=300)
    plt.close()
    
    # Create a detailed comparison table plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Method', 'Avg Loss', 'Avg Coverage', 'Avg Time (s)'])
    for method in ['rl', 'bho', 'pso']:
        table_data.append([
            method.upper(),
            f"{np.mean(results[method]['losses']):.4f} ± {np.std(results[method]['losses']):.4f}",
            f"{np.mean(results[method]['coverages']):.4f} ± {np.std(results[method]['coverages']):.4f}",
            f"{np.mean(results[method]['times']):.2f} ± {np.std(results[method]['times']):.2f}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Detailed Method Comparison', fontsize=16, pad=20)
    plt.savefig(f"{save_dir}/comparison_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Run comparison
    results = compare_methods(n_maps=10, map_size=64)
    
    print("\nComparison complete! Results saved to results/comparison/")