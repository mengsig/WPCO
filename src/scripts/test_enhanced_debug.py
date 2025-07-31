#!/usr/bin/env python3
"""Debug script for enhanced training."""

import os
import sys

# Set CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.scripts.enhanced_train import train_enhanced_agent

if __name__ == "__main__":
    try:
        # Run with debug mode and fewer episodes
        agent, best_coverage = train_enhanced_agent(episodes=100, debug=True)
        print(f"\nTest completed successfully! Best coverage: {best_coverage:.1%}")
    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()