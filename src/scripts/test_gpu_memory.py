import torch
import numpy as np
import sys
import os

# Append the utils directory to the system path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.algorithms.rl_agent_parallel import ImprovedDQNAgent, CirclePlacementEnv

def test_gpu_memory():
    """Test if the model fits in GPU memory."""
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Free GPU Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")
    else:
        print("No GPU available, using CPU")
    
    print("\nTesting model creation and forward pass...")
    
    try:
        # Create agent
        agent = ImprovedDQNAgent(
            map_size=64,
            batch_size=32,
            buffer_size=10000  # Smaller buffer for testing
        )
        
        print("✓ Agent created successfully")
        
        # Create a batch of states
        batch_size = 32
        map_size = 64
        states = np.random.rand(batch_size, map_size * map_size + 2).astype(np.float32)
        
        # Test forward pass
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(agent.device)
            output = agent.q_network(states_tensor)
            print(f"✓ Forward pass successful. Output shape: {output.shape}")
        
        # Test training step
        env = CirclePlacementEnv(map_size=64)
        state = env.reset()
        
        # Simulate some experiences
        print("\nSimulating experiences...")
        for i in range(100):
            action = agent.act(state, env.get_valid_actions_mask())
            next_state, reward, done, info = env.step(action)
            agent.remember_n_step(state, action, reward, next_state, done)
            
            if done:
                state = env.reset()
            else:
                state = next_state
        
        print(f"✓ Added {len(agent.memory)} experiences to replay buffer")
        
        # Test replay
        if len(agent.memory) >= agent.batch_size:
            print("\nTesting replay training...")
            for i in range(5):
                agent.replay()
                print(f"✓ Replay step {i+1} completed")
        
        if torch.cuda.is_available():
            print(f"\nFinal GPU Memory Usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        
        print("\n✅ All tests passed! The model should work with your GPU.")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if "out of memory" in str(e).lower():
            print("\nSuggestions to reduce memory usage:")
            print("1. Reduce batch_size (currently 32)")
            print("2. Reduce hidden_size in the network")
            print("3. Use gradient accumulation")
            print("4. Enable mixed precision training")

if __name__ == "__main__":
    test_gpu_memory()