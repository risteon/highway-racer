# RACER Risk Metrics Documentation

This document provides a detailed explanation of how risk metrics work in the RACER (Risk-sensitive Actor Critic with Epistemic Robustness) implementation across different simulation environments.

## Overview

RACER implements safety-critical reinforcement learning through distributional Q-learning with CVaR (Conditional Value at Risk) risk measures. The risk metrics serve as safety signals that guide the agent toward risk-sensitive policies for autonomous driving applications.

## Risk Metric Implementations

### 1. MuJoCo Simulation Environment (Procedural Driving)

**Location**: `scripts/sim/train_online_states.py:86-107`

**Risk Type**: Vehicle Rollover Detection

**What it measures**: Vehicle orientation stability to detect rollover events

#### Implementation:
```python
def safety_reward_fn(next_obs):
    quat = next_obs["car/body_rotation"]  # Vehicle quaternion rotation
    quat = npq.as_quat_array(quat)
    up_world = np.array([0, 0, 1])        # World "up" vector (Z-axis)
    up_local = npq.as_vector_part(         # Vehicle's local "up" vector
        quat * npq.from_vector_part(up_world) * np.conjugate(quat)
    )
    error = np.linalg.norm(up_local - up_world, axis=-1) / np.sqrt(2)
    dot_product = np.sum(up_local * up_world, axis=-1)
    
    # Binary penalty: -1 if upside down, 0 otherwise
    rewards = np.where(dot_product < 0, -1, 0)
    return rewards
```

#### How it works:
- Computes the angle between vehicle's "up" direction and world "up" direction using quaternions
- **Binary detection**: If dot product < 0, vehicle is upside down → penalty = -1.0
- Otherwise, no penalty (0.0)
- Focuses on **intrinsic vehicle safety** - preventing complete rollover failures

#### Characteristics:
- **Type**: Binary (discrete failure detection)
- **Sensitivity**: Only triggers on complete rollover (dot product < 0)
- **Range**: {-1.0, 0.0}
- **Purpose**: Prevent catastrophic vehicle orientation failures

### 2. Highway-Env Environment

**Location**: `scripts/sim/train_highway_states.py:93-137`

**Risk Type**: Collision Proximity Risk

**What it measures**: Collision risk based on distance to other vehicles

#### Implementation:
```python
def safety_reward_fn(obs, info=None):
    obs = obs.reshape(15, 6)  # [presence, x, y, vx, vy, heading]
    present_vehicles = obs[obs[:, 0] > 0.5]  # Filter present vehicles
    
    if len(present_vehicles) <= 1:
        return 0.0  # Only ego vehicle, no collision risk
    
    ego_vehicle = present_vehicles[0]
    other_vehicles = present_vehicles[1:]
    
    ego_pos = ego_vehicle[1:3]  # [x, y] position
    other_pos = other_vehicles[:, 1:3]
    
    distances = np.linalg.norm(other_pos - ego_pos, axis=1)
    min_distance = np.min(distances)
    
    # Tiered penalty system
    safety_threshold = 10.0    # meters - safe following distance
    collision_threshold = 2.0  # meters - collision imminent
    
    if min_distance < collision_threshold:
        collision_risk = -1.0  # Maximum penalty for imminent collision
    elif min_distance < safety_threshold:
        # Exponential decay penalty for unsafe proximity
        collision_risk = -np.exp(-(min_distance - collision_threshold) / 
                                (safety_threshold - collision_threshold))
    else:
        collision_risk = 0.0   # Safe distance, no penalty
    return collision_risk
```

#### How it works:
- Monitors distances to all other vehicles in the 15-vehicle observation space
- **Three-tier penalty system**:
  1. **< 2m**: Maximum penalty (-1.0) - imminent collision
  2. **2m-10m**: Exponential decay penalty - smooth risk gradient
  3. **> 10m**: No penalty (0.0) - safe following distance
- Focuses on **external interaction safety** - preventing vehicle-to-vehicle collisions

#### Characteristics:
- **Type**: Continuous (smooth risk gradient)
- **Sensitivity**: High sensitivity in 2-10m proximity range
- **Range**: [-1.0, 0.0] with exponential decay
- **Purpose**: Maintain safe following distances and prevent collisions

#### Risk Parameters:
- `collision_threshold = 2.0m`: Distance below which maximum penalty is applied
- `safety_threshold = 10.0m`: Distance above which no penalty is applied
- **Exponential decay**: Smooth transition between thresholds using `exp(-(d-2)/(10-2))`

## Integration in Training Loop

Both risk metrics are integrated identically into the RACER training pipeline:

### Step 1: Compute Safety Reward
```python
safety_bonus = safety_reward_fn(next_observation, info)
```

### Step 2: Modify Environment Reward
```python
reward += safety_bonus * safety_bonus_coeff
```
- **MuJoCo**: `safety_bonus_coeff` not explicitly set (defaults to 0.0)
- **Highway-Env**: `safety_bonus_coeff = 0.1` (moderate weighting)

### Step 3: Store in Replay Buffer
```python
replay_buffer.insert(dict(
    observations=observation,
    actions=action,
    rewards=reward,
    masks=mask,
    dones=done,
    next_observations=next_observation,
    safety=-safety_bonus,  # Negated for safety critic training
))
```

### Step 4: Update Safety Components
```python
if hasattr(agent, "update_safety"):
    agent, safety_info = agent.update_safety(-safety_ema)
    update_info = {**update_info, **safety_info}
```

### Step 5: Logging and Monitoring
```python
safety_ema = (1-ema_beta) * safety_ema + ema_beta * safety_bonus
wandb.log({"safety_ema": safety_ema}, step=i)
```

## Dual Purpose in RACER Algorithm

The risk metrics serve **two critical functions** in RACER:

### 1. Immediate Reward Shaping
- **Purpose**: Guide policy learning during environment interaction
- **Mechanism**: Modifies the immediate reward signal to penalize unsafe behaviors
- **Effect**: Encourages the agent to avoid risky states and actions in real-time

### 2. Safety Critic Training (for safety-aware agents)
- **Purpose**: Train separate safety critic networks for risk-sensitive updates
- **Mechanism**: Uses the `safety` field in replay buffer for safety-specific learning
- **Effect**: Enables CVaR-based policy updates and adaptive action limit learning

## Comparison: MuJoCo vs Highway-Env

| Aspect | MuJoCo (Rollover) | Highway-Env (Collision) |
|--------|------------------|-------------------------|
| **Risk Type** | Physical stability | Inter-vehicle collision |
| **Continuity** | Binary (0 or -1) | Continuous (exponential) |
| **Sensitivity** | Complete rollover only | Proximity-sensitive (2-10m) |
| **Detection Range** | Single threshold | Multi-tier system |
| **Safety Focus** | Intrinsic vehicle safety | External interaction safety |
| **Gradient** | None (discrete) | Smooth exponential decay |
| **Coefficient** | 0.0 (disabled) | 0.1 (moderate) |

## Risk Metric Selection Guidelines

### Choose MuJoCo Rollover Metric when:
- Focus is on **vehicle dynamics** and stability
- Binary failure detection is sufficient
- Testing **catastrophic failure prevention**
- Studying intrinsic vehicle safety limits

### Choose Highway-Env Collision Metric when:
- Focus is on **multi-agent interactions**
- Continuous risk assessment is needed
- Testing **collision avoidance** behaviors
- Studying **risk-sensitive policy learning** with smooth gradients

## Configuration

### MuJoCo Environment
```python
# In distributional_limits_config.py
config.safety_penalty = 0.0  # Typically disabled
```

### Highway-Env Environment
```python
# In highway_distributional_config.py
config.safety_penalty = 0.1  # Moderate safety weighting
```

## Training Commands

### MuJoCo/Procedural Training:
```bash
cd /home/risteon/workspace/gpudrive_docker/racer
source /home/risteon/miniconda3/bin/activate racer
DISPLAY=:0 WANDB_MODE=offline python scripts/sim/train_online_states.py \
  --config scripts/sim/configs/distributional_limits_config.py \
  --world_name flat
```

### Highway-Env Training:
```bash
cd /home/risteon/workspace/gpudrive_docker/racer
source /home/risteon/miniconda3/bin/activate racer
DISPLAY=:0 WANDB_MODE=offline python scripts/sim/train_highway_states.py \
  --config scripts/sim/configs/highway_distributional_config.py \
  --max_steps 100000
```

## Research Applications

The risk metrics enable research in:

1. **Risk-Sensitive Reinforcement Learning**: Using CVaR objectives with continuous risk signals
2. **Safety-Critical Control**: Studying safe policy learning in autonomous driving
3. **Distributional RL**: Analyzing tail risk behavior in safety-critical domains
4. **Sim-to-Real Transfer**: Validating safety-aware policies on real robotic systems

## Implementation Notes

- **Observation Space**: Highway-env uses flattened 90D observations (15 vehicles × 6 features)
- **Action Space**: Both environments use continuous action spaces (2D for highway, variable for MuJoCo)
- **Gymnasium Compatibility**: Highway-env requires gymnasium instead of legacy gym
- **Safety Critic**: Available in `safety_critic_sac` and `safety_sac_learner` agent variants

The highway-env implementation provides a more sophisticated continuous risk metric that better captures the nuanced nature of collision risk in multi-agent driving scenarios, making it particularly well-suited for studying risk-sensitive reinforcement learning algorithms like RACER.