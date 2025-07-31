# Massive Simulation Configuration

## Overview
The enhanced training script is now configured for a 2 million episode simulation with optimized exploration strategy.

## Key Configuration

### Workers
- **64 workers** (up from 32) for faster parallel experience collection
- Automatically detects available cores and uses up to 64

### Epsilon Decay Strategy (Based on EPISODES, not steps)
- **Phase 1 (0-500k episodes)**: Heavy exploration
  - Linear decay: ε = 1.0 → 0.5
  - ~25% of total training dedicated to exploration
  
- **Phase 2 (500k-1.5M episodes)**: Balanced exploration/exploitation
  - Exponential decay: ε = 0.5 → 0.05
  - ~50% of training for learning optimal strategies
  
- **Phase 3 (1.5M-2M episodes)**: Fine-tuning
  - Slow linear decay: ε = 0.05 → 0.01
  - ~25% of training for exploitation and refinement

### Saving Strategy
- **Model checkpoints**: Every 100,000 episodes (20 total)
- **Visualizations**: Every 5,000 episodes (400 total)
- **Progress updates**: Every 10,000 episodes (200 total)

### Reward Normalization
- Rewards are scaled down by 100 and clipped to [-10, 10]
- Prevents Q-value explosion and stabilizes training

## Usage

```bash
# Run 2 million episodes (default)
python src/scripts/enhanced_random_dqn_train.py

# Run custom number of episodes
python src/scripts/enhanced_random_dqn_train.py --episodes 1000000

# Specify workers
python src/scripts/enhanced_random_dqn_train.py --workers 32
```

## Important Notes

1. **Epsilon is based on EPISODES**: The epsilon decay is tied to episode count, not training steps. With 64 workers, episodes complete much faster than training steps.

2. **Training steps vs Episodes**: 
   - Episodes = Complete circle placement sequences
   - Training steps = Neural network updates
   - With async training, these are decoupled

3. **Expected Timeline**:
   - Phase 1 (exploration): ~8-10 hours
   - Phase 2 (learning): ~16-20 hours  
   - Phase 3 (fine-tuning): ~8-10 hours
   - Total: ~32-40 hours (depending on hardware)

4. **Memory Usage**: With 1M replay buffer and 64 workers, expect ~10-15GB RAM usage

5. **GPU Usage**: The neural network training uses GPU while workers use CPU