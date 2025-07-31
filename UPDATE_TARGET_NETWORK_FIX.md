# Fix for 'GuidedDQNAgent' object has no attribute 'update_target_network' Error

## Problem
The training script was failing with the error:
```
Training step error: 'GuidedDQNAgent' object has no attribute 'update_target_network'
```

This occurred because the `random_dqn_train.py` script was calling `self.agent.update_target_network()` every `target_update_freq` steps (default: 1000), but the `GuidedDQNAgent` class didn't have this method.

## Root Cause
The `GuidedDQNAgent` was using soft updates (with tau=0.001) in its `replay()` method, but the training script expected a hard update method `update_target_network()` to be available.

## Solution
Added two methods to the `GuidedDQNAgent` class:

1. **`update_target_network()`** - Performs a hard update (complete copy) of weights from Q-network to target network
2. **`soft_update_target_network()`** - Performs a soft update using tau parameter (already existed inline, now extracted to a method)

## Code Changes

### In `src/algorithms/dqn_agent.py`:

```python
def update_target_network(self):
    """Update the target network with current Q-network weights."""
    self.target_network.load_state_dict(self.q_network.state_dict())

def soft_update_target_network(self):
    """Perform soft update of target network parameters."""
    for target_param, param in zip(
        self.target_network.parameters(), self.q_network.parameters()
    ):
        target_param.data.copy_(
            self.tau * param.data + (1.0 - self.tau) * target_param.data
        )
```

## Update Strategy
The agent now supports both update strategies:
- **Soft updates**: Applied after every training step in `replay()` with tau=0.001
- **Hard updates**: Applied every 1000 steps via `update_target_network()`

This hybrid approach can help with stability:
- Soft updates provide smooth, gradual target network updates
- Periodic hard updates ensure the target network doesn't lag too far behind

## Testing
Created `test_update_target_fix.py` to verify:
- The method exists
- It can be called without errors
- Both update methods are available

The error should no longer occur during training.