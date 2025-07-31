#!/usr/bin/env python3
"""Test script to demonstrate robust periodic tracking."""

import sys
import os
import time
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.periodic_tracker import RobustPeriodicChecker, PeriodicTaskTracker


def test_robust_checker():
    """Test the RobustPeriodicChecker with skipped steps."""
    print("Testing RobustPeriodicChecker with skipped steps...")
    print("=" * 60)
    
    # Create a checker that should execute every 100 steps
    checker = RobustPeriodicChecker(interval=100)
    
    # Simulate asynchronous updates with random skips
    current_step = 0
    executions = []
    
    for _ in range(50):
        # Random step increment (simulating async behavior)
        step_increment = random.randint(1, 50)
        current_step += step_increment
        
        if checker.should_execute(current_step):
            executions.append(current_step)
            print(f"✓ Executed at step {current_step}")
    
    print(f"\nTotal executions: {len(executions)}")
    print(f"Execution steps: {executions}")
    
    # Check that we didn't miss any 100-step intervals
    expected_executions = current_step // 100
    print(f"\nExpected executions: {expected_executions}")
    print(f"Actual executions: {len(executions)}")
    print(f"Success: {'✓' if len(executions) >= expected_executions else '✗'}")


def test_modulo_failure():
    """Demonstrate how modulo-based checks can fail."""
    print("\n\nDemonstrating modulo-based failure...")
    print("=" * 60)
    
    # Simulate the exact scenario from the user's issue
    save_interval = 1000
    current_step = 0
    saves = []
    
    # Simulate steps that skip the exact multiple
    steps = [997, 1003, 1999, 2001, 3005, 4002, 5001, 6003, 7002, 8001, 9037]
    
    print("Using modulo-based checking (if step % 1000 == 0):")
    for step in steps:
        current_step = step
        if current_step % save_interval == 0:
            saves.append(current_step)
            print(f"  Saved at step {current_step}")
    
    print(f"\nModulo-based saves: {saves}")
    print(f"First save at: {saves[0] if saves else 'Never!'}")
    
    # Now show robust checker handling the same scenario
    print("\n\nUsing RobustPeriodicChecker:")
    checker = RobustPeriodicChecker(interval=save_interval)
    saves_robust = []
    
    for step in steps:
        if checker.should_execute(step):
            saves_robust.append(step)
            print(f"  Saved at step {step}")
    
    print(f"\nRobust saves: {saves_robust}")
    print(f"First save at: {saves_robust[0] if saves_robust else 'Never'}")


def test_periodic_task_tracker():
    """Test the PeriodicTaskTracker with multiple tasks."""
    print("\n\nTesting PeriodicTaskTracker with multiple tasks...")
    print("=" * 60)
    
    tracker = PeriodicTaskTracker()
    
    # Track executions
    saves = []
    updates = []
    cleanups = []
    
    # Register tasks
    tracker.register_task('save', 500, lambda step: saves.append(step))
    tracker.register_task('update', 100, lambda step: updates.append(step))
    tracker.register_task('cleanup', 250, lambda step: cleanups.append(step))
    
    # Simulate irregular steps
    current_step = 0
    for _ in range(30):
        step_increment = random.randint(20, 80)
        current_step += step_increment
        
        executed = tracker.check_and_execute(current_step)
        for task, was_executed in executed.items():
            if was_executed:
                print(f"  Step {current_step}: Executed {task}")
    
    print(f"\nSave executions: {saves}")
    print(f"Update executions: {updates}")
    print(f"Cleanup executions: {cleanups}")
    
    # Show status
    status = tracker.get_status()
    print("\nTask Status:")
    for name, info in status.items():
        print(f"  {name}: {info['executions']} executions, next at step {info['next_target']}")


if __name__ == "__main__":
    test_robust_checker()
    test_modulo_failure()
    test_periodic_task_tracker()
    
    print("\n\n✓ All tests completed!")
    print("\nKey benefits of robust periodic tracking:")
    print("1. Never misses intervals due to skipped step counts")
    print("2. Handles asynchronous updates gracefully")
    print("3. Tracks multiple periodic tasks efficiently")
    print("4. Thread-safe for concurrent environments")