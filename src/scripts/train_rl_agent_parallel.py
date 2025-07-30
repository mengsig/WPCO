import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time
from collections import deque

# Append the utils directory to the system path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.plotting_utils import (
    plot_defaults,
    plot_heatmap_with_circles,
)
from src.algorithms.rl_agent_parallel import (
    ImprovedDQNAgent, CirclePlacementEnv, random_seeder
)


def run_parallel_episode(env, agent, episode_num):
    """Run a single episode and return results."""
    # Generate a new random map
    weighted_matrix = random_seeder(env.map_size, time_steps=100000)
    state = env.reset(weighted_matrix)
    
    episode_reward = 0
    done = False
    steps = 0
    
    while not done:
        # Get valid actions mask
        valid_mask = env.get_valid_actions_mask()
        
        # Choose action
        action = agent.act(state, valid_mask)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Store experience with n-step
        agent.remember_n_step(state, action, reward, next_state, done)
        
        # Update state
        state = next_state
        episode_reward += reward
        steps += 1
    
    return {
        'episode': episode_num,
        'reward': episode_reward,
        'coverage': info['coverage_ratio'],
        'steps': steps,
        'placements': env.placed_circles.copy(),
        'original_map': env.original_map.copy()
    }


def train_rl_agent_parallel(n_episodes=5000, map_size=64, radii=None, 
                           save_interval=100, n_parallel=4,
                           train_interval=4, train_steps=10):
    """
    Train the RL agent with parallel environment simulation.
    """
    if radii is None:
        radii = [12, 8, 7, 6, 5, 4, 3, 2, 1]
    
    # Initialize agent with improved hyperparameters
    agent = ImprovedDQNAgent(
        map_size=map_size,
        radii=radii,
        learning_rate=1e-4,  # Reduced for stability
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=int(n_episodes * 0.8),  # Decay over 80% of episodes for better exploration
        buffer_size=50000,  # Smaller buffer for faster learning
        batch_size=32,
        tau=0.001,  # Much smaller for stability
        n_step=1,  # Single step for stability
        use_double_dqn=True
    )
    
    # Create environments for parallel execution
    envs = [CirclePlacementEnv(map_size=map_size, radii=radii) for _ in range(n_parallel)]
    
    # Training metrics
    episode_rewards = []
    episode_coverage = []
    episode_steps = []
    best_coverage = 0
    best_placements = None
    best_map = None
    
    # Moving averages for smooth metrics
    reward_window = deque(maxlen=100)
    coverage_window = deque(maxlen=100)
    steps_window = deque(maxlen=100)
    
    print(f"Training RL agent for {n_episodes} episodes with {n_parallel} parallel environments...")
    print(f"Epsilon will decay from {agent.epsilon_start} to {agent.epsilon_end} over {agent.epsilon_decay_steps} steps")
    print(f"Using improved architecture with attention mechanisms and prioritized replay")
    print("-" * 80)
    
    # Progress bar
    pbar = tqdm(total=n_episodes, desc="Training")
    
    # Training loop
    episode = 0
    while episode < n_episodes:
        # Run parallel episodes
        with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            # Submit parallel episodes
            futures = []
            for i in range(min(n_parallel, n_episodes - episode)):
                future = executor.submit(run_parallel_episode, envs[i], agent, episode + i)
                futures.append(future)
            
            # Collect results
            for future in futures:
                result = future.result()
                
                # Update metrics
                episode_rewards.append(result['reward'])
                episode_coverage.append(result['coverage'])
                episode_steps.append(result['steps'])
                
                # Update moving averages
                reward_window.append(result['reward'])
                coverage_window.append(result['coverage'])
                steps_window.append(result['steps'])
                
                # Track best performance
                if result['coverage'] > best_coverage:
                    best_coverage = result['coverage']
                    best_placements = result['placements']
                    best_map = result['original_map']
                
                episode += 1
                pbar.update(1)
        
        # Train the agent
        if len(agent.memory) > agent.batch_size and episode % train_interval == 0:
            for _ in range(train_steps):
                agent.replay()
        
        # Print detailed statistics
        if episode % save_interval == 0 and episode > 0:
            stats = agent.get_statistics()
            
            print(f"\n{'='*80}")
            print(f"Episode {episode}/{n_episodes}")
            print(f"{'='*80}")
            
            # Performance metrics
            print(f"\nPERFORMANCE METRICS:")
            print(f"  Coverage - Current: {episode_coverage[-1]:.4f}, "
                  f"Avg (last 100): {np.mean(list(coverage_window)):.4f}, "
                  f"Best: {best_coverage:.4f}")
            print(f"  Reward - Current: {episode_rewards[-1]:.2f}, "
                  f"Avg (last 100): {np.mean(list(reward_window)):.2f}")
            print(f"  Steps - Current: {episode_steps[-1]}, "
                  f"Avg (last 100): {np.mean(list(steps_window)):.1f}")
            
            # Training metrics
            print(f"\nTRAINING METRICS:")
            print(f"  Epsilon: {stats['epsilon']:.4f}")
            print(f"  Learning Rate: {stats['learning_rate']:.6f}")
            print(f"  Avg Loss: {stats['avg_loss']:.4f}")
            print(f"  Avg Q-Value: {stats['avg_q_value']:.4f}")
            print(f"  Avg Gradient Norm: {stats['avg_gradient_norm']:.4f}")
            print(f"  Replay Buffer Size: {stats['buffer_size']:,}")
            
            # Improvement tracking
            if len(episode_coverage) >= 200:
                recent_avg = np.mean(episode_coverage[-100:])
                previous_avg = np.mean(episode_coverage[-200:-100])
                improvement = ((recent_avg - previous_avg) / previous_avg) * 100
                print(f"\nIMPROVEMENT:")
                print(f"  Coverage improvement over last 100 episodes: {improvement:+.2f}%")
            
            print(f"{'='*80}\n")
            
            # Save checkpoint
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'episode': episode,
                'model_state_dict': agent.q_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'best_coverage': best_coverage,
                'episode_rewards': episode_rewards,
                'episode_coverage': episode_coverage,
            }, f"{checkpoint_dir}/improved_rl_agent_checkpoint_{episode}.pt")
    
    pbar.close()
    
    # Final statistics
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nFINAL STATISTICS:")
    print(f"  Total Episodes: {n_episodes}")
    print(f"  Best Coverage Achieved: {best_coverage:.4f}")
    print(f"  Final Avg Coverage (last 100): {np.mean(episode_coverage[-100:]):.4f}")
    print(f"  Final Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"  Final Epsilon: {agent.epsilon:.4f}")
    print(f"  Total Training Time: {pbar.format_dict['elapsed']:.1f} seconds")
    print(f"{'='*80}\n")
    
    return agent, episode_rewards, episode_coverage, best_placements, best_map


def evaluate_agent(agent, n_eval_episodes=50, map_size=64, radii=None):
    """
    Evaluate the trained agent on new maps with detailed metrics.
    """
    if radii is None:
        radii = [12, 8, 7, 6, 5, 4, 3, 2, 1]
    
    env = CirclePlacementEnv(map_size=map_size, radii=radii)
    
    eval_rewards = []
    eval_coverage = []
    eval_steps = []
    placement_positions = []
    
    best_eval_coverage = 0
    best_eval_placements = None
    best_eval_map = None
    
    # Set agent to evaluation mode (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    print(f"\n{'='*80}")
    print(f"EVALUATION PHASE")
    print(f"{'='*80}")
    print(f"Evaluating agent on {n_eval_episodes} new maps...")
    
    for episode in tqdm(range(n_eval_episodes), desc="Evaluating"):
        # Generate a new random map
        weighted_matrix = random_seeder(map_size, time_steps=100000)
        state = env.reset(weighted_matrix)
        
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Get valid actions mask
            valid_mask = env.get_valid_actions_mask()
            
            # Choose action (greedy)
            action = agent.act(state, valid_mask)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Update state
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Record metrics
        eval_rewards.append(episode_reward)
        coverage = info['coverage_ratio']
        eval_coverage.append(coverage)
        eval_steps.append(steps)
        placement_positions.append(env.placed_circles.copy())
        
        # Track best performance
        if coverage > best_eval_coverage:
            best_eval_coverage = coverage
            best_eval_placements = env.placed_circles.copy()
            best_eval_map = env.original_map.copy()
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    # Calculate detailed statistics
    coverage_stats = {
        'mean': np.mean(eval_coverage),
        'std': np.std(eval_coverage),
        'min': np.min(eval_coverage),
        'max': np.max(eval_coverage),
        'median': np.median(eval_coverage),
        'percentile_25': np.percentile(eval_coverage, 25),
        'percentile_75': np.percentile(eval_coverage, 75)
    }
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"\nCOVERAGE STATISTICS:")
    print(f"  Mean ± Std: {coverage_stats['mean']:.4f} ± {coverage_stats['std']:.4f}")
    print(f"  Median (IQR): {coverage_stats['median']:.4f} "
          f"({coverage_stats['percentile_25']:.4f} - {coverage_stats['percentile_75']:.4f})")
    print(f"  Min - Max: {coverage_stats['min']:.4f} - {coverage_stats['max']:.4f}")
    
    print(f"\nOTHER METRICS:")
    print(f"  Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"  Average Steps: {np.mean(eval_steps):.1f} ± {np.std(eval_steps):.1f}")
    
    # Consistency analysis
    coverage_cv = np.std(eval_coverage) / np.mean(eval_coverage)
    print(f"\nCONSISTENCY:")
    print(f"  Coefficient of Variation: {coverage_cv:.4f}")
    print(f"  Performance Rating: ", end="")
    if coverage_cv < 0.05:
        print("Excellent (very consistent)")
    elif coverage_cv < 0.10:
        print("Good (consistent)")
    elif coverage_cv < 0.15:
        print("Fair (somewhat consistent)")
    else:
        print("Needs improvement (inconsistent)")
    
    print(f"{'='*80}\n")
    
    return eval_rewards, eval_coverage, best_eval_placements, best_eval_map, coverage_stats


def plot_enhanced_training_progress(episode_rewards, episode_coverage, agent, save_dir="results"):
    """
    Create enhanced visualization of training progress.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Define grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Moving average window
    window = 100
    
    # 1. Coverage over time (main metric)
    ax1.plot(episode_coverage, alpha=0.3, label='Raw coverage', color='blue')
    if len(episode_coverage) >= window:
        moving_avg = np.convolve(episode_coverage, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_coverage)), moving_avg, 
                label=f'{window}-episode moving average', linewidth=2, color='darkblue')
    ax1.axhline(y=np.max(episode_coverage), color='red', linestyle='--', alpha=0.5, label='Best coverage')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Coverage Ratio')
    ax1.set_title('Coverage Performance Over Training', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rewards
    ax2.plot(episode_rewards, alpha=0.3, color='green')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_rewards)), moving_avg, 
                linewidth=2, color='darkgreen')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Rewards Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning metrics
    if agent.losses:
        ax3.semilogy(agent.losses, alpha=0.5, color='orange')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss (log scale)')
        ax3.set_title('Training Loss')
        ax3.grid(True, alpha=0.3)
    
    # 4. Q-values
    if agent.q_values:
        ax4.plot(agent.q_values, alpha=0.5, color='purple')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Max Q-Value')
        ax4.set_title('Q-Value Evolution')
        ax4.grid(True, alpha=0.3)
    
    # 5. Coverage distribution histogram
    ax5.hist(episode_coverage[-1000:], bins=30, alpha=0.7, color='teal', edgecolor='black')
    ax5.axvline(x=np.mean(episode_coverage[-1000:]), color='red', linestyle='--', 
                label=f'Mean: {np.mean(episode_coverage[-1000:]):.3f}')
    ax5.set_xlabel('Coverage Ratio')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Coverage Distribution (Last 1000 Episodes)')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Enhanced Training Progress Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/enhanced_training_progress.png", dpi=300)
    plt.close()
    
    # Create epsilon decay plot
    fig, ax = plt.subplots(figsize=(8, 5))
    episodes = list(range(0, len(episode_coverage), 10))
    epsilons = []
    temp_agent = ImprovedDQNAgent(epsilon_decay_steps=agent.epsilon_decay_steps)
    for ep in episodes:
        temp_agent.steps_done = ep
        temp_agent.update_epsilon()
        epsilons.append(temp_agent.epsilon)
    
    ax.plot(episodes, epsilons, linewidth=2)
    ax.axvline(x=agent.steps_done, color='red', linestyle='--', 
               label=f'Current: ε={agent.epsilon:.3f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (Epsilon) Decay Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epsilon_decay.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_defaults()
    
    # Training parameters
    n_episodes = 5000  # Increased for better convergence
    n_eval_episodes = 50  # More evaluation episodes
    map_size = 64
    radii = [12, 8, 7, 6, 5, 4, 3, 2, 1]
    
    # Set random seed for reproducibility
    seed = np.random.randint(10000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print(f"Running with seed: {seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Train the agent with parallelization
    start_time = time.time()
    agent, episode_rewards, episode_coverage, best_placements, best_map = train_rl_agent_parallel(
        n_episodes=n_episodes,
        map_size=map_size,
        radii=radii,
        save_interval=100,
        n_parallel=min(4, mp.cpu_count() // 2),  # Limit to 4 for GPU memory
        train_interval=4,
        train_steps=10
    )
    training_time = time.time() - start_time
    
    print(f"\nTotal training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    print(f"Time per episode: {training_time/n_episodes:.2f} seconds")
    
    # Evaluate the trained agent
    eval_rewards, eval_coverage, best_eval_placements, best_eval_map, coverage_stats = evaluate_agent(
        agent, 
        n_eval_episodes=n_eval_episodes,
        map_size=map_size,
        radii=radii
    )
    
    # Create results directory
    results_dir = "results/rl_agent_improved"
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot enhanced training progress
    plot_enhanced_training_progress(episode_rewards, episode_coverage, agent, save_dir=results_dir)
    
    # Visualize best solutions
    if best_placements is not None and best_map is not None:
        # Convert placements to position array
        pos = []
        for x, y, r in best_placements:
            pos.extend([x, y])
        pos = np.array(pos)
        
        plot_heatmap_with_circles(
            best_map, pos, radii,
            savename=f"{results_dir}/best_training_solution.png"
        )
    
    if best_eval_placements is not None and best_eval_map is not None:
        # Convert placements to position array
        pos = []
        for x, y, r in best_eval_placements:
            pos.extend([x, y])
        pos = np.array(pos)
        
        plot_heatmap_with_circles(
            best_eval_map, pos, radii,
            savename=f"{results_dir}/best_eval_solution.png"
        )
    
    # Save final model
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'episode_rewards': episode_rewards,
        'episode_coverage': episode_coverage,
        'coverage_stats': coverage_stats,
        'training_time': training_time,
        'n_episodes': n_episodes,
    }, f"{results_dir}/final_improved_model.pt")
    
    print(f"\nAll results saved to {results_dir}/")
    
    # Create a summary report
    with open(f"{results_dir}/training_summary.txt", 'w') as f:
        f.write("REINFORCEMENT LEARNING AGENT TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  Episodes: {n_episodes}\n")
        f.write(f"  Map Size: {map_size}x{map_size}\n")
        f.write(f"  Radii: {radii}\n")
        f.write(f"  Random Seed: {seed}\n")
        f.write(f"  Training Time: {training_time:.1f} seconds\n\n")
        
        f.write(f"Training Results:\n")
        f.write(f"  Best Coverage (Training): {max(episode_coverage):.4f}\n")
        f.write(f"  Final Avg Coverage (last 100): {np.mean(episode_coverage[-100:]):.4f}\n")
        f.write(f"  Final Epsilon: {agent.epsilon:.4f}\n\n")
        
        f.write(f"Evaluation Results (on {n_eval_episodes} new maps):\n")
        f.write(f"  Mean Coverage: {coverage_stats['mean']:.4f} ± {coverage_stats['std']:.4f}\n")
        f.write(f"  Median Coverage: {coverage_stats['median']:.4f}\n")
        f.write(f"  Best Coverage: {coverage_stats['max']:.4f}\n")
        f.write(f"  Consistency (CV): {np.std(eval_coverage) / np.mean(eval_coverage):.4f}\n")