# Intelligent Saving Implementation

## Overview

Both `enhanced_train_async.py` and `random_dqn_train.py` now implement intelligent saving that automatically distributes saves evenly across the training duration.

## Key Features

### 1. **Automatic Frequency Calculation**
Instead of fixed intervals, save frequencies are calculated based on total episodes:
- **Model Checkpoints**: ~20 saves distributed evenly (e.g., every 5,000 episodes for 100k total)
- **Visualizations**: ~100 saves distributed evenly (e.g., every 1,000 episodes for 100k total)
- **Progress Updates**: ~50 evaluations distributed evenly

### 2. **Organized Directory Structure**
All saves are organized into directories:
- `checkpoints/` - Model checkpoint files (.pth)
- `visualizations/` - Training progress images (.png)

### 3. **Informative Filenames**
Files are named with zero-padded episode numbers for proper sorting:
- Models: `enhanced_async_ep000500_cov45%.pth`
- Images: `progress_ep001000.png`

### 4. **Configuration Display**
Training scripts now show save configuration at startup:
```
Save Configuration:
  - Model checkpoints: ~20 saves (every 5000 episodes)
  - Visualizations: ~100 saves (every 1000 episodes)
  - Progress updates: ~50 times (every 2000 episodes)
  - All saves in: checkpoints/ and visualizations/ directories
```

## Implementation Details

### Enhanced Train Async
```python
# Calculate save frequencies based on total episodes
total_episodes = config.get('episodes', 100000)
n_model_saves = config.get('n_model_saves', 20)
n_image_saves = config.get('n_image_saves', 100)

model_save_freq = max(100, total_episodes // n_model_saves)
image_save_freq = max(100, total_episodes // n_image_saves)
```

### Visualization Content
Each visualization includes:
- Coverage over episodes (with 37% plateau reference line)
- Rewards over episodes
- Training loss (log scale)
- Coverage distribution histogram

## Cleanup Utility

A utility script `cleanup_saves.py` helps manage saved files:

```bash
# List all saved files
python cleanup_saves.py list

# Organize files into directories
python cleanup_saves.py organize

# Clean up old saves (keep 20 models, 100 images)
python cleanup_saves.py cleanup

# Do everything automatically
python cleanup_saves.py auto
```

## Benefits

1. **Disk Space Efficiency**: No more thousands of checkpoint files
2. **Better Organization**: Easy to find specific saves
3. **Consistent Coverage**: Saves distributed evenly across training
4. **Flexibility**: Easy to adjust number of saves via config
5. **Automatic Management**: Set and forget approach

## Usage

### For Enhanced Async Training:
```python
config = {
    'episodes': 100000,
    'n_model_saves': 20,    # Total model checkpoints to save
    'n_image_saves': 100,   # Total visualizations to save
}
```

### For Random DQN Training:
The same configuration is automatically applied based on the total number of episodes.

## Storage Estimates

For a typical 100k episode run:
- Model checkpoints: ~20 files × ~50MB = ~1GB
- Visualizations: ~100 files × ~2MB = ~200MB
- Total: ~1.2GB (vs 10-20GB without intelligent saving)

This implementation ensures efficient storage usage while maintaining good coverage of the training process for analysis and model selection.