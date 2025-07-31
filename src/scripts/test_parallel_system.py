#!/usr/bin/env python3
"""
Simple test script for the parallel training system.
Tests basic functionality without requiring external dependencies.
"""

import sys
import os
import multiprocessing as mp
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch available")
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch not available: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy available")
        print(f"   Version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy not available: {e}")
        return False
    
    try:
        import matplotlib
        print("✅ Matplotlib available")
    except ImportError as e:
        print(f"❌ Matplotlib not available: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("✅ tqdm available")
    except ImportError as e:
        print(f"❌ tqdm not available: {e}")
        return False
    
    try:
        from src.algorithms.dqn_agent import GuidedDQNAgent, AdvancedCirclePlacementEnv
        print("✅ DQN Agent modules available")
    except ImportError as e:
        print(f"❌ DQN Agent modules not available: {e}")
        return False
    
    return True

def test_multiprocessing():
    """Test multiprocessing functionality."""
    print("\n🔄 Testing multiprocessing...")
    
    def worker_test(worker_id, result_queue):
        """Simple worker function for testing."""
        result_queue.put(f"Worker {worker_id} completed")
    
    try:
        # Test with a few processes
        n_workers = min(4, mp.cpu_count())
        result_queue = mp.Queue()
        workers = []
        
        print(f"   Starting {n_workers} test workers...")
        
        for i in range(n_workers):
            worker = mp.Process(target=worker_test, args=(i, result_queue))
            worker.start()
            workers.append(worker)
        
        # Collect results
        results = []
        for _ in range(n_workers):
            try:
                result = result_queue.get(timeout=5)
                results.append(result)
            except:
                print("❌ Worker timeout")
                return False
        
        # Wait for workers to finish
        for worker in workers:
            worker.join(timeout=2)
            if worker.is_alive():
                worker.terminate()
        
        print(f"✅ Multiprocessing test passed")
        print(f"   Results: {results}")
        return True
        
    except Exception as e:
        print(f"❌ Multiprocessing test failed: {e}")
        return False

def test_basic_training():
    """Test basic training components."""
    print("\n🏋️ Testing basic training components...")
    
    try:
        from src.algorithms.dqn_agent import GuidedDQNAgent, AdvancedCirclePlacementEnv, random_seeder
        import torch
        
        # Create small test environment
        print("   Creating test environment...")
        env = AdvancedCirclePlacementEnv(map_size=32)  # Small for testing
        
        # Create test agent
        print("   Creating test agent...")
        agent = GuidedDQNAgent(
            map_size=32,
            learning_rate=1e-3,
            batch_size=4,
            buffer_size=100,
        )
        
        # Test one episode
        print("   Running test episode...")
        test_map = random_seeder(32, time_steps=1000)  # Small for testing
        state = env.reset(test_map)
        
        steps = 0
        while steps < 10:  # Limited steps for testing
            valid_mask = env.get_valid_actions_mask()
            action = agent.act(state, env, valid_mask)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state if not done else None, done)
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Test training step
        if len(agent.memory) > agent.batch_size:
            print("   Testing training step...")
            loss = agent.replay()
            print(f"   Training loss: {loss}")
        
        print("✅ Basic training test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 PARALLEL TRAINING SYSTEM TEST")
    print("=" * 50)
    
    # System info
    print(f"Python version: {sys.version}")
    print(f"CPU cores: {mp.cpu_count()}")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("Multiprocessing Test", test_multiprocessing),
        ("Basic Training Test", test_basic_training),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour system is ready for parallel training!")
        print("\nNext steps:")
        print("1. Install missing dependencies if any were reported")
        print("2. Run: python3 src/scripts/parallel_train_dqn.py")
        print("3. Or use the ultra version: python3 src/scripts/ultra_parallel_train_dqn.py")
        return 0
    else:
        print("💥 SOME TESTS FAILED!")
        print("\nPlease fix the issues above before running parallel training.")
        return 1

if __name__ == "__main__":
    exit(main())