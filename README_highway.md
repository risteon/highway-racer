# RACER Highway Environment

This document describes the highway-env integration for RACER (Risk-sensitive Actor Critic with Epistemic Robustness).

## Setup

First, set up the racer environment:
```bash
conda create -n racer python=3.11
conda activate racer
cd /home/risteon/workspace/gpudrive_docker/racer
pip install -e jaxrl5[train] dmcgym .
pip install -r requirements.txt
```

## Running Highway Training

```bash
conda activate racer
cd /home/risteon/workspace/gpudrive_docker/racer
WANDB_MODE=offline python scripts/sim/train_highway_states.py \
  --config scripts/sim/configs/highway_distributional_config.py \
  --max_steps 100000
```

## Highway Environment Specifications

### Observation Space
- **Type**: Flattened kinematics observations
- **Shape**: `(90,)` - flattened from `(15, 6)`
- **Features**: 15 vehicles × 6 features each:
  1. **presence** - binary indicator if vehicle exists
  2. **x** - longitudinal position 
  3. **y** - lateral position
  4. **vx** - longitudinal velocity
  5. **vy** - lateral velocity  
  6. **heading** - vehicle orientation
- **Bounds**: `[-100.0, 100.0]` (wrapped for replay buffer compatibility)

### Action Space
- **Type**: `ContinuousAction`
- **Shape**: `(2,)` 
- **Actions**:
  1. **steering** - lateral control `[-1, 1]`
  2. **acceleration** - longitudinal control `[-1, 1]`
- **Adaptive Limits**: Steering has learnable bounds (initial: 0.2) for conservative control

### Environment Configuration
- **Vehicles**: 50 total, 15 observed
- **Lanes**: 4-lane highway
- **Episode**: 40 seconds, max 1000 steps  
- **Frequencies**: 15Hz simulation, 5Hz policy
- **Safety**: Collision threshold 2.0m, safe distance 10.0m

### Risk-Sensitive Features
- **CVaR Risk**: 0.8 (high safety sensitivity)
- **Q-value Range**: [-50, 50] with 101 atoms for distributional learning
- **Safety Penalty**: Exponential decay based on proximity to other vehicles

## Reward Calculation

RACER uses a composite training reward that addresses highway-env limitations and adds safety considerations. The reward is implemented in `highway_safety_utils.calculate_training_reward()` and used consistently across training and evaluation.

### Training Reward Components

The final training reward combines three components:

```python
training_reward = base_reward - original_speed * speed_weight + forward_speed * speed_weight + safety_reward * safety_coeff
```

#### 1. Base Environment Reward

Highway-env calculates reward from four weighted components:

**Component Calculation**:
```python
# Individual reward components (all in [0,1] except collision)
collision_reward = float(vehicle.crashed)           # {0, 1}
right_lane_reward = lane_index / max(lanes-1, 1)   # [0, 1] 
high_speed_reward = clip(speed_ratio, 0, 1)        # [0, 1]
on_road_reward = float(vehicle.on_road)             # {0, 1}

# Final weighted combination
base_reward = (collision_weight × collision_reward + 
               right_lane_weight × right_lane_reward +
               speed_weight × high_speed_reward +
               lane_change_weight × lane_change_action) × on_road_reward
```

**Default Highway-v0 Weights**:
- `collision_reward: -1.0` - Strong crash penalty
- `right_lane_reward: 0.1` - Small right-lane preference  
- `high_speed_reward: 0.4` - Medium speed incentive
- `lane_change_reward: 0.0` - No lane change penalty
- `reward_speed_range: [20, 30]` - Optimal speed range (m/s)
- `normalize_reward: True` - Normalize to [0,1] range

**Configurable Parameters**:
All weights can be customized via environment config:
```python
highway_config = {
    "collision_reward": -1.0,     # Crash penalty weight
    "right_lane_reward": 0.1,     # Right lane preference weight  
    "high_speed_reward": 0.4,     # Speed reward weight
    "lane_change_reward": 0.0,    # Lane change penalty weight
    "reward_speed_range": [20, 30], # Speed range for linear mapping
    "normalize_reward": True      # Whether to normalize final reward
}
```

**Range**: Base reward ∈ [-1.0, 0.5] before normalization, [0, 1] after normalization

#### 2. Forward Speed Correction
- **Problem**: Highway-env rewards backward driving in speed component
- **Solution**: Replace with forward-only speed reward
- **Calculation**: 
  ```python
  forward_speed = vehicle.velocity[0]  # Only longitudinal component
  speed_reward = (forward_speed - min_speed) / (max_speed - min_speed)
  speed_reward = clip(speed_reward, 0.0, 1.0)  # No reward for backward driving
  ```
- **Parameters**: 
  - `reward_speed_range = [30, 45]` m/s (optimal speed range)
  - `speed_weight = 0.4` (highway-env default weight)

#### 3. Safety Reward
Combines collision risk assessment and offroad detection:

**Collision Risk** (Time-to-Collision based):
- **Predictive**: Uses TTC instead of just distance
- **Directional**: Only considers vehicles in forward driving cone (±11.25°)
- **Progressive**: Exponential penalties based on collision urgency
- **Dual Constraints**: TTC < 2.0s AND distance < 1.0m for activation
- **Range**: [0.0, -1.0] (most negative = immediate danger)

**Offroad Penalty**:
- **Enhanced Detection**: Uses lateral distance from lane center
- **Method**: More accurate than highway-env's permissive `on_road` property  
- **Calculation**: `abs(lateral_position) > lane_width/2`
- **Penalty**: Fixed -0.5 when offroad

**Safety Coefficient**: `safety_penalty = 0.01` (configurable, small but important)

### Reward Ranges

| Component | Range | Typical Value |
|-----------|-------|---------------|
| Base Reward | [-1, 1] | ~0.7 |
| Speed Correction | [-0.4, 0.4] | ~0.2 |
| Safety Reward | [-0.5, 0.0] | ~-0.05 |
| **Total Training** | **[-1.9, 1.4]** | **~0.85** |

### Implementation Consistency

All three scripts use the same reward calculation:

1. **Training Loop** (`train_highway_states.py`): Uses `calculate_training_reward()` for policy updates
2. **Online Evaluation** (`run_trajectory()` in training): Uses same function for consistent eval  
3. **Policy Evaluation** (`run_highway_policy.py`): Uses same function for fair assessment

### Key Parameters

```python
# In highway_distributional_config.py
safety_penalty = 0.01           # Safety reward weight
reward_speed_range = [30, 45]   # Optimal speed range (m/s)  
speed_weight = 0.4              # Highway-env speed component weight
```

### Reward Design Rationale

1. **Forward-Only Speed**: Prevents unrealistic backward driving policies
2. **Predictive Safety**: TTC-based collision avoidance vs reactive distance-only
3. **Enhanced Offroad**: More accurate lane boundary detection  
4. **Risk Sensitivity**: CVaR training focuses on worst-case scenarios
5. **Consistency**: Same reward across training/evaluation prevents distribution shift

## Policy Evaluation

Evaluate trained highway policies using saved checkpoints:

```bash
conda activate racer
cd /home/risteon/workspace/gpudrive_docker/racer

# Basic evaluation
python scripts/sim/run_highway_policy.py \
  --policy_file policies/highway_run_42/checkpoint_10000 \
  --config scripts/sim/configs/highway_distributional_config.py \
  --num_episodes 20 \
  --video_output_dir ./evaluation_videos

# Evaluation without video rendering (faster)
python scripts/sim/run_highway_policy.py \
  --policy_file policies/highway_run_42/checkpoint_10000 \
  --config scripts/sim/configs/highway_distributional_config.py \
  --num_episodes 50 \
  --render=false \
  --video_output_dir ./evaluation_results
```

### Evaluation Outputs

The evaluation script generates:

- **Performance Metrics**:
  - Mean episode return and standard deviation
  - Success rate (episodes without collision)
  - Safety violation rates
  - Average minimum distance to other vehicles
  - Average ego vehicle speed

- **Video Files** (if `--render=true`):
  - MP4 videos showing agent behavior
  - Text overlays with real-time metrics
  - Saved as `highway_eval_{checkpoint}_episode_{i}.mp4`

- **Detailed Results** (`evaluation_metrics.json`):
  - Per-episode statistics
  - Aggregated summary metrics
  - Configuration details

### Example Output

```
==================================================
HIGHWAY POLICY EVALUATION RESULTS
==================================================
Episodes: 20
Mean Return: 24.567 ± 8.234
Mean Length: 187.3 ± 45.2
Success Rate: 85.0%
Collision Rate: 15.0%
Mean Safety Violation Rate: 12.3%
Mean Min Distance: 8.45m
Mean Ego Speed: 22.1 m/s
```

## Key Files
- `scripts/sim/train_highway_states.py` - Main highway training script
- `scripts/sim/run_highway_policy.py` - Policy evaluation script
- `scripts/sim/configs/highway_distributional_config.py` - Highway-specific RACER configuration
- Uses `highway-v0` environment with continuous action control and collision risk safety rewards