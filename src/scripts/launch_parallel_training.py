#!/usr/bin/env python3
"""
Parallel DQN Training Launcher
Automatically detects system capabilities and launches optimal training configuration.
"""

import sys
import os
import argparse
import psutil
import torch
import subprocess
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'psutil': 'psutil',
        'scipy': 'SciPy (optional, for advanced features)'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(name)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall missing packages with:")
        print("pip install torch numpy matplotlib tqdm psutil scipy")
        return False
    
    print("✅ All dependencies available")
    return True

def detect_system_capabilities():
    """Detect system capabilities and recommend training mode."""
    n_cores = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    gpu_memory = 0
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print("🖥️  SYSTEM CAPABILITIES")
    print("=" * 50)
    print(f"CPU Cores: {n_cores}")
    print(f"RAM: {memory_gb:.1f} GB")
    print(f"GPU: {'✅' if gpu_available else '❌'}")
    if gpu_available:
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("=" * 50)
    
    # Recommend training mode
    if n_cores >= 64 and memory_gb >= 32:
        if gpu_available and gpu_memory >= 8:
            recommended = "ultra"
            print("🚀 RECOMMENDATION: Ultra-Parallel Training")
            print("   Your system can handle maximum performance mode!")
        else:
            recommended = "parallel"
            print("⚡ RECOMMENDATION: Parallel Training")
            print("   Great CPU setup, but limited GPU - using standard parallel mode")
    elif n_cores >= 16 and memory_gb >= 16:
        recommended = "parallel"
        print("⚡ RECOMMENDATION: Parallel Training")
        print("   Good system for parallel training")
    else:
        recommended = "original"
        print("📚 RECOMMENDATION: Original Training")
        print("   Limited resources - using single-threaded training")
    
    return {
        'cores': n_cores,
        'memory_gb': memory_gb,
        'gpu_available': gpu_available,
        'gpu_memory': gpu_memory,
        'recommended': recommended
    }

def run_training(mode, episodes=None, workers=None, batch_size=None):
    """Launch training with specified mode and parameters."""
    print(f"\n🎯 LAUNCHING {mode.upper()} TRAINING")
    print("=" * 50)
    
    if mode == "original":
        # Run original training script
        script_path = "src/scripts/train_dqn.py"
        cmd = [sys.executable, script_path]
        
    elif mode == "parallel":
        # Run parallel training script
        script_path = "src/scripts/parallel_train_dqn.py"
        cmd = [sys.executable, script_path]
        
    elif mode == "ultra":
        # Run ultra-parallel training script
        script_path = "src/scripts/ultra_parallel_train_dqn.py"
        cmd = [sys.executable, script_path]
    
    else:
        print(f"❌ Unknown training mode: {mode}")
        return False
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        return False
    
    print(f"📄 Script: {script_path}")
    print(f"🚀 Starting training...")
    print("=" * 50)
    
    try:
        # Launch training
        result = subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="Launch parallel DQN training with optimal configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  original  - Single-threaded training (for limited resources)
  parallel  - Multi-process parallel training (recommended for 16+ cores)
  ultra     - Ultra-high performance training (for 64+ cores, GPU recommended)
  auto      - Automatically detect best mode (default)

Examples:
  python launch_parallel_training.py                    # Auto-detect best mode
  python launch_parallel_training.py --mode ultra       # Force ultra mode
  python launch_parallel_training.py --episodes 10000   # Custom episode count
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["original", "parallel", "ultra", "auto"],
        default="auto",
        help="Training mode to use (default: auto)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of episodes to train (overrides script defaults)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes (for parallel modes)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check system and dependencies, don't start training"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip system capability checks and use specified mode"
    )
    
    args = parser.parse_args()
    
    print("🤖 PARALLEL DQN TRAINING LAUNCHER")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before continuing.")
        return 1
    
    # Detect system capabilities
    system_info = detect_system_capabilities()
    
    # Determine training mode
    if args.mode == "auto":
        mode = system_info['recommended']
        print(f"\n🎯 Auto-selected mode: {mode}")
    else:
        mode = args.mode
        if not args.force and mode != system_info['recommended']:
            print(f"\n⚠️  WARNING: You selected '{mode}' but '{system_info['recommended']}' is recommended")
            print("Use --force to skip this warning")
            
            response = input("Continue anyway? (y/N): ").lower()
            if response != 'y':
                print("Training cancelled.")
                return 0
    
    if args.check_only:
        print(f"\n✅ System check complete. Recommended mode: {system_info['recommended']}")
        return 0
    
    # Launch training
    success = run_training(mode, args.episodes, args.workers, args.batch_size)
    
    if success:
        print("\n🎉 Training session completed!")
        
        # Show some helpful next steps
        print("\n📋 NEXT STEPS:")
        print("1. Check the generated plots and model files")
        print("2. Evaluate your trained model on test environments")
        print("3. Consider running additional training with different hyperparameters")
        
        return 0
    else:
        print("\n💥 Training failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())