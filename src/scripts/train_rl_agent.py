import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Append the utils directory to the system path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.plotting_utils import (
    plot_defaults,
    plot_losses,
    plot_heatmap_with_circles,
)
from src.algorithms.rl_agent import DQNAgent, CirclePlacementEnv, random_seeder


def train_rl_agent(n_episodes=1000, map_size=64, radii=None, 
                   save_interval=100, update_target_interval=10):
    """
    Train the RL agent on multiple randomly generated maps.
    """
    if radii is None:
        radii = [12, 8, 7, 6, 5, 4, 3, 2, 1]
    
    # Initialize agent and environment
    agent = DQNAgent(map_size=map_size, radii=radii)
    env = CirclePlacementEnv(map_size=map_size, radii=radii)
    
    # Training metrics
    episode_rewards = []
    episode_coverage = []
    best_coverage = 0
    best_placements = None
    best_map = None
    
    print(f"Training RL agent for {n_episodes} episodes...")
    
    for episode in tqdm(range(n_episodes)):
        # Generate a new random map for each episode
        weighted_matrix = random_seeder(map_size, time_steps=100000)
        state = env.reset(weighted_matrix)
        
        episode_reward = 0
        done = False
        
        while not done:
            # Get valid actions mask
            valid_mask = env.get_valid_actions_mask()
            
            # Choose action
            action = agent.act(state, valid_mask)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            
            # Train the agent
            if len(agent.memory) > agent.batch_size:
                agent.replay()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        coverage = info['coverage_ratio']
        episode_coverage.append(coverage)
        
        # Track best performance
        if coverage > best_coverage:
            best_coverage = coverage
            best_placements = env.placed_circles.copy()
            best_map = env.original_map.copy()
        
        # Update target network
        if episode % update_target_interval == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Save checkpoint and visualize progress
        if episode % save_interval == 0 and episode > 0:
            print(f"\nEpisode {episode}:")
            print(f"  Average reward: {np.mean(episode_rewards[-100:]):.2f}")
            print(f"  Average coverage: {np.mean(episode_coverage[-100:]):.2f}")
            print(f"  Best coverage so far: {best_coverage:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            
            # Save model checkpoint
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'episode': episode,
                'model_state_dict': agent.q_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'best_coverage': best_coverage,
            }, f"{checkpoint_dir}/rl_agent_checkpoint_{episode}.pt")
    
    return agent, episode_rewards, episode_coverage, best_placements, best_map


def evaluate_agent(agent, n_eval_episodes=10, map_size=64, radii=None):
    """
    Evaluate the trained agent on new maps.
    """
    if radii is None:
        radii = [12, 8, 7, 6, 5, 4, 3, 2, 1]
    
    env = CirclePlacementEnv(map_size=map_size, radii=radii)
    
    eval_rewards = []
    eval_coverage = []
    best_eval_coverage = 0
    best_eval_placements = None
    best_eval_map = None
    
    # Set agent to evaluation mode (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    print(f"\nEvaluating agent on {n_eval_episodes} new maps...")
    
    for episode in range(n_eval_episodes):
        # Generate a new random map
        weighted_matrix = random_seeder(map_size, time_steps=100000)
        state = env.reset(weighted_matrix)
        
        episode_reward = 0
        done = False
        
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
        
        # Record metrics
        eval_rewards.append(episode_reward)
        coverage = info['coverage_ratio']
        eval_coverage.append(coverage)
        
        # Track best performance
        if coverage > best_eval_coverage:
            best_eval_coverage = coverage
            best_eval_placements = env.placed_circles.copy()
            best_eval_map = env.original_map.copy()
        
        print(f"  Episode {episode + 1}: Coverage = {coverage:.3f}")
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    print(f"\nEvaluation Results:")
    print(f"  Average coverage: {np.mean(eval_coverage):.3f}")
    print(f"  Best coverage: {best_eval_coverage:.3f}")
    
    return eval_rewards, eval_coverage, best_eval_placements, best_eval_map


def plot_training_progress(episode_rewards, episode_coverage, save_dir="results"):
    """
    Plot training progress including rewards and coverage over episodes.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot rewards
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Moving average window
    window = 100
    
    # Plot episode rewards
    ax1.plot(episode_rewards, alpha=0.3, label='Raw rewards')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                label=f'{window}-episode moving average', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot coverage ratio
    ax2.plot(episode_coverage, alpha=0.3, label='Raw coverage')
    if len(episode_coverage) >= window:
        moving_avg = np.convolve(episode_coverage, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_coverage)), moving_avg, 
                label=f'{window}-episode moving average', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Coverage Ratio')
    ax2.set_title('Coverage Ratio Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/rl_training_progress.png", dpi=300)
    plt.close()


def visualize_best_solution(weighted_matrix, placements, radii, title, savename):
    """
    Visualize the best solution found.
    """
    # Convert placements to the format expected by plot_heatmap_with_circles
    pos = []
    for i, (x, y, r) in enumerate(placements):
        pos.extend([x, y])
    
    pos = np.array(pos)
    plot_heatmap_with_circles(weighted_matrix, pos, radii, savename=savename)
    
    # Also create a version showing the coverage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original map
    ax1.imshow(weighted_matrix, cmap='hot', interpolation='nearest')
    ax1.set_title('Original Weighted Map')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Map with circles
    ax2.imshow(weighted_matrix, cmap='hot', interpolation='nearest')
    from matplotlib.patches import Circle
    for x, y, r in placements:
        circle = Circle((x, y), radius=r, color='blue', alpha=0.5)
        ax2.add_patch(circle)
    ax2.set_title(f'{title}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_xlim(0, weighted_matrix.shape[1])
    ax2.set_ylim(weighted_matrix.shape[0], 0)
    
    plt.tight_layout()
    plt.savefig(savename.replace('.png', '_comparison.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_defaults()
    
    # Training parameters
    n_episodes = 2000  # Number of training episodes
    n_eval_episodes = 20  # Number of evaluation episodes
    map_size = 64
    radii = [12, 8, 7, 6, 5, 4, 3, 2, 1]
    
    # Set random seed for reproducibility
    seed = np.random.randint(10000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Running with seed: {seed}")
    
    # Train the agent
    agent, episode_rewards, episode_coverage, best_placements, best_map = train_rl_agent(
        n_episodes=n_episodes,
        map_size=map_size,
        radii=radii,
        save_interval=200,
        update_target_interval=10
    )
    
    # Evaluate the trained agent
    eval_rewards, eval_coverage, best_eval_placements, best_eval_map = evaluate_agent(
        agent, 
        n_eval_episodes=n_eval_episodes,
        map_size=map_size,
        radii=radii
    )
    
    # Create results directory
    results_dir = "results/rl_agent"
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot training progress
    plot_training_progress(episode_rewards, episode_coverage, save_dir=results_dir)
    
    # Visualize best solutions
    if best_placements is not None and best_map is not None:
        visualize_best_solution(
            best_map, 
            best_placements, 
            radii,
            f"Best Training Solution (Coverage: {max(episode_coverage):.3f})",
            f"{results_dir}/best_training_solution.png"
        )
    
    if best_eval_placements is not None and best_eval_map is not None:
        visualize_best_solution(
            best_eval_map, 
            best_eval_placements, 
            radii,
            f"Best Evaluation Solution (Coverage: {max(eval_coverage):.3f})",
            f"{results_dir}/best_eval_solution.png"
        )
    
    # Save final model
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'episode_rewards': episode_rewards,
        'episode_coverage': episode_coverage,
    }, f"{results_dir}/final_model.pt")
    
    print(f"\nTraining complete! Results saved to {results_dir}/")
    print(f"Final average coverage (last 100 episodes): {np.mean(episode_coverage[-100:]):.3f}")
    print(f"Best coverage achieved during training: {max(episode_coverage):.3f}")
    print(f"Average evaluation coverage: {np.mean(eval_coverage):.3f}")