#!/usr/bin/env python3
"""Test script to verify enhanced radii context functionality."""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.dqn_agent import AdvancedCirclePlacementEnv
from utils.map_generation import random_seeder


def test_radii_context():
    """Test the enhanced radii context features."""
    print("Testing Enhanced Radii Context...")
    
    # Create environment with custom radii
    custom_radii = [20, 18, 15, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1]
    env = AdvancedCirclePlacementEnv(map_size=64, radii=custom_radii)
    
    # Generate a test map
    test_map = random_seeder(64, time_steps=50000)
    state = env.reset(test_map)
    
    print("\n1. Initial State Analysis:")
    print(f"   - Feature vector size: {len(state['features'])}")
    print(f"   - All radii encoding shape: {state['all_radii_encoding'].shape}")
    print(f"   - Placement status shape: {state['placement_status'].shape}")
    
    # Visualize initial context
    env.visualize_radii_context()
    
    print("\n2. Feature Vector Breakdown:")
    features = state['features']
    print(f"   Base features (0-9):")
    print(f"     - Current radius: {features[0]:.3f}")
    print(f"     - Progress: {features[1]:.3f}")
    print(f"     - Num clusters: {features[2]:.3f}")
    print(f"     - Max potential: {features[3]:.3f}")
    print(f"     - Coverage: {features[4]:.3f}")
    print(f"     - Can fit count: {features[5]:.3f}")
    print(f"     - Avg density: {features[6]:.3f}")
    print(f"     - Remaining circles: {features[7]:.3f}")
    print(f"     - Largest remaining: {features[8]:.3f}")
    print(f"     - Space utilization: {features[9]:.3f}")
    
    print(f"\n   Radii values (10-29):")
    for i in range(20):
        if i < len(custom_radii):
            print(f"     - Radius {i+1}: {features[10+i]:.3f} (actual: {custom_radii[i]})")
        else:
            print(f"     - Radius {i+1}: {features[10+i]:.3f} (padding)")
    
    print(f"\n   Placement status (30-49):")
    for i in range(20):
        if i < len(custom_radii):
            status = "Placed" if features[30+i] == 1.0 else "Not placed"
            print(f"     - Circle {i+1}: {status}")
    
    print("\n3. Testing Future Placement Analysis:")
    # Test placing first circle at center
    test_x, test_y = 32, 32
    future_analysis = env.analyze_future_placement_potential(test_x, test_y, custom_radii[0])
    
    print(f"   Placing circle at ({test_x}, {test_y}) with radius {custom_radii[0]}:")
    print(f"     - Future circles that can fit: {future_analysis['future_fit_count']}/{len(custom_radii)-1}")
    print(f"     - Future placement ratio: {future_analysis['future_placement_ratio']:.3f}")
    print(f"     - Blocked area: {future_analysis['blocked_area']:.1f}")
    print(f"     - Blocked area ratio: {future_analysis['blocked_area_ratio']:.3f}")
    print(f"     - Placement efficiency: {future_analysis['placement_efficiency']:.3f}")
    
    # Test placing at corner
    test_x2, test_y2 = 20, 20
    future_analysis2 = env.analyze_future_placement_potential(test_x2, test_y2, custom_radii[0])
    
    print(f"\n   Placing circle at ({test_x2}, {test_y2}) with radius {custom_radii[0]}:")
    print(f"     - Future circles that can fit: {future_analysis2['future_fit_count']}/{len(custom_radii)-1}")
    print(f"     - Future placement ratio: {future_analysis2['future_placement_ratio']:.3f}")
    print(f"     - Blocked area: {future_analysis2['blocked_area']:.1f}")
    print(f"     - Blocked area ratio: {future_analysis2['blocked_area_ratio']:.3f}")
    print(f"     - Placement efficiency: {future_analysis2['placement_efficiency']:.3f}")
    
    print("\n4. Simulating a few placements:")
    # Place first circle
    action = (32, 32)
    state, reward, done, info = env.step(action)
    print(f"\n   After placing circle 1:")
    print(f"     - Reward: {reward:.3f}")
    print(f"     - Coverage: {info['coverage']:.1%}")
    
    # Place second circle
    if not done:
        action = (32, 52)
        state, reward, done, info = env.step(action)
        print(f"\n   After placing circle 2:")
        print(f"     - Reward: {reward:.3f}")
        print(f"     - Coverage: {info['coverage']:.1%}")
        print(f"     - Feature vector still has size: {len(state['features'])}")
    
    # Visualize updated context
    env.visualize_radii_context()
    
    print("\n5. State Dictionary Keys:")
    if state:
        print(f"   Available keys: {list(state.keys())}")
        print(f"   Radii statistics: {state['radii_statistics']}")
    
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test_radii_context()