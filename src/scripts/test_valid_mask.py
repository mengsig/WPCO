import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from algorithms.rl_agent_parallel import CirclePlacementEnv, random_seeder

# Create environment
env = CirclePlacementEnv(map_size=64)
weighted_matrix = random_seeder(64, time_steps=100000)
state = env.reset(weighted_matrix)

print("Testing Valid Actions Mask...")
print("=" * 50)

# Place first circle
print("\n1. Placing first circle (radius=12)...")
valid_mask = env.get_valid_actions_mask()
valid_positions = np.argwhere(valid_mask)
print(f"   Valid positions: {len(valid_positions)} out of {64*64}")

# Place in center
action = (32, 32)
state, reward, done, info = env.step(action)
print(f"   Placed at {action}, reward: {reward:.3f}")

# Check second circle
print("\n2. Checking valid positions for second circle (radius=8)...")
valid_mask = env.get_valid_actions_mask()
valid_positions = np.argwhere(valid_mask)
print(f"   Valid positions: {len(valid_positions)} out of {64*64}")

# Check if center is now invalid
if not valid_mask[32, 32]:
    print("   ✓ Center position correctly marked as invalid!")
else:
    print("   ✗ ERROR: Center position still marked as valid!")

# Check positions near the first circle
print("\n3. Checking positions around first circle...")
min_distance = 12 + 8  # radius1 + radius2
for dx in [-10, -15, -20, -25]:
    x, y = 32 + dx, 32
    dist = abs(dx)
    is_valid = valid_mask[x, y]
    expected = dist >= min_distance
    status = "✓" if is_valid == expected else "✗"
    print(f"   Position ({x}, {y}), distance={dist}: valid={is_valid} {status}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original map
axes[0].imshow(env.original_map, cmap='hot')
axes[0].set_title('Original Map')
axes[0].axis('off')

# Current map with circle
axes[1].imshow(env.current_map, cmap='hot')
circle = plt.Circle((32, 32), 12, fill=False, color='blue', linewidth=2)
axes[1].add_patch(circle)
axes[1].set_title('After First Circle')
axes[1].axis('off')

# Valid mask
axes[2].imshow(valid_mask, cmap='RdYlGn')
axes[2].set_title('Valid Positions for Next Circle (Green=Valid)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('valid_mask_test.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to valid_mask_test.png")

# Test with multiple circles
print("\n4. Testing with multiple circles...")
env.reset(weighted_matrix)
positions = [(20, 20), (40, 40), (20, 40), (40, 20)]
for i, pos in enumerate(positions[:3]):
    state, reward, done, info = env.step(pos)
    print(f"   Placed circle {i+1} at {pos}")

valid_mask = env.get_valid_actions_mask()
valid_count = np.sum(valid_mask)
print(f"   Valid positions remaining: {valid_count}")

# Check if all placed positions are invalid
all_invalid = True
for px, py, pr in env.placed_circles:
    if valid_mask[int(px), int(py)]:
        all_invalid = False
        print(f"   ✗ ERROR: Position ({px}, {py}) with circle is still valid!")

if all_invalid:
    print("   ✓ All circle positions correctly marked as invalid!")

print("\nTest complete!")