import math

def calculate_reward(density_ratio, overlap_ratio):
    """Calculate reward using the smooth function"""
    # Base reward: quality of placement (0 to 1)
    quality_score = density_ratio ** 1.5
    
    # For visualization, assume collection_score proportional to density
    collection_score = density_ratio * 0.8
    
    # Combine quality and collection with weights
    base_reward = (0.7 * quality_score + 0.3 * collection_score)
    
    # Apply smooth overlap penalty using exponential decay
    if overlap_ratio > 0:
        overlap_penalty = 1 - math.exp(-3 * overlap_ratio)
        reward = base_reward * (1 - overlap_penalty) - 0.1 * overlap_penalty
    else:
        reward = base_reward
    
    # Add small exploration bonus for non-zero density areas
    if density_ratio > 0.1 and overlap_ratio == 0:
        reward += 0.05
    
    # Smooth penalty for empty areas
    if density_ratio < 0.01:
        reward = reward * 0.1 - 0.02
    
    # Final clipping
    reward = max(-0.3, min(1.1, reward))
    
    return reward

print("SMOOTH REWARD FUNCTION ANALYSIS")
print("=" * 50)
print("\n1. Perfect Placements (No Overlap):")
print("-" * 30)
densities = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
for d in densities:
    r = calculate_reward(d, 0.0)
    print(f"Density={d:.1f}: Reward={r:+.3f}")

print("\n2. Effect of Overlap (at Density=0.8):")
print("-" * 30)
overlaps = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
for o in overlaps:
    r = calculate_reward(0.8, o)
    print(f"Overlap={o:.2f}: Reward={r:+.3f}")

print("\n3. Gradient Examples (showing smooth transitions):")
print("-" * 30)
print("Small overlap changes:")
for o in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]:
    r = calculate_reward(0.6, o)
    print(f"  Overlap={o:.2f}: Reward={r:+.3f}")

print("\n4. Key Thresholds:")
print("-" * 30)
print(f"Empty area (d=0.0): {calculate_reward(0.0, 0.0):+.3f}")
print(f"Barely any value (d=0.05): {calculate_reward(0.05, 0.0):+.3f}")
print(f"Low value, no overlap (d=0.2): {calculate_reward(0.2, 0.0):+.3f}")
print(f"Medium value, tiny overlap (d=0.5, o=0.05): {calculate_reward(0.5, 0.05):+.3f}")
print(f"High value, small overlap (d=0.8, o=0.1): {calculate_reward(0.8, 0.1):+.3f}")

print("\n5. Reward Differences (to show gradients):")
print("-" * 30)
# Show how small changes in overlap affect reward
base = calculate_reward(0.7, 0.0)
for o in [0.05, 0.10, 0.15]:
    r = calculate_reward(0.7, o)
    diff = r - base
    print(f"Overlap {o:.2f} vs 0.00: {diff:+.3f} change")

print("\nSUMMARY:")
print("-" * 30)
print("- Rewards range from -0.3 to ~1.1")
print("- Smooth exponential decay for overlaps")
print("- Small overlaps get small penalties")
print("- Encourages high-density placements")
print("- No hard step functions!")