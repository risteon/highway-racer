# CLAUDE.md

This file provides guidance to Claude Code when working with the RACER repository implementation.

## Project Overview

**RACER** (Risk-sensitive Actor Critic with Epistemic Robustness) is a research implementation for offroad autonomous driving that addresses safety-critical control through distributional reinforcement learning. The repository implements the algorithm described in the paper "RACER: Risk-sensitive Actor Critic with Epistemic Robustness" by Kyle Stachowicz and Sergey Levine.

**Key Innovation**: Uses distributional Q-learning with CVaR (Conditional Value at Risk) risk measures to learn risk-sensitive policies for safe offroad navigation.

## Repository Structure

The codebase is organized into four main components:

### 1. Core RL Framework (`jaxrl5/`)
JAX-based implementations of various reinforcement learning algorithms optimized for continuous control:

- **Distributional SAC** (`jaxrl5/agents/distributional_sac/`) - Core RACER algorithm
- **Safety-Critical SAC** (`jaxrl5/agents/safety_critic_sac/`) - Safety-aware variants
- **Gaussian Distributional SAC** (`jaxrl5/agents/gaussian_distributional_sac/`) - Alternative distributions
- **Standard Algorithms** - SAC, TD3, Quantile networks, MPPI

### 2. Simulation Environment (`offroad-procedural-sim/`)
Procedural offroad driving simulation built on MuJoCo/dm_control:

- **Procedural Arena** (`procedural_arena.py`) - Infinite terrain generation
- **Car Dynamics** (`car.py`) - Vehicle physics and control
- **Task Definition** (`task.py`) - Goal-oriented navigation with reward functions
- **World Configurations** (`configs/worlds/`) - Terrain presets (flat, bumpy, default)

### 3. Real Robot Integration (`offroad-robot-ros/`)
ROS-based system for deploying learned policies on real hardware:

- **Data Collection** (`offroad_learning/`) - Training data aggregation
- **Inference** (`inference/`) - Real-time policy execution
- **Sensor Fusion** (`offroad_sensor_fusion/`) - GPS and IMU integration
- **Hardware Interface** - VESC motor control, camera integration

### 4. Training Scripts (`scripts/sim/`)
Training configurations and execution scripts:

- **Algorithm Configs** (`configs/`) - Hyperparameters for different algorithms
- **Training Pipeline** (`train_online_states.py`) - Main training script
- **Evaluation Tools** (`run_policy.py`, `manual.py`) - Policy testing

## Algorithm Implementations

### 1. Distributional Epistemic Critic
**Location**: `jaxrl5/agents/distributional_sac/distributional_agent.py:33-55`

**Implementation**: `DistributionalStateActionValue` class
- **Distributional Q-Networks**: Lines 33-55 output probability distributions over returns instead of scalar Q-values
- **Categorical Distribution**: Uses `num_atoms` (default: 151) discrete atoms spanning `[q_min, q_max]` ([-100, 650])
- **Epistemic Uncertainty**: Ensemble of Q-networks (`networks/ensemble.py:8-37`) provides uncertainty quantification
- **Target Computation**: Lines 102-145 implement distributional Bellman backup with categorical projection

**Key Functions**:
- `distributional_target()` (Lines 102-145): Categorical projection for distributional backup
- `update_distributional_critic()` (Lines 147-280): Main critic training with ensemble reduction

### 2. Risk-Sensitive Actor (CVaR)
**Location**: `jaxrl5/agents/distributional_sac/distributional_agent.py:316-324`

**Implementation**: `cvar()` function
```python
def cvar(probs, atoms, risk) -> jax.Array:
    cdf = jnp.cumsum(probs, axis=-1)
    cdf_clipped = jnp.clip(cdf, a_min=0, a_max=1 - risk)
    pdf_clipped = jnp.diff(...) / (1 - risk)
    return jnp.sum(pdf_clipped * atoms, axis=-1)
```

**Actor Training** (Lines 327-375):
- **Risk-Sensitive Objective**: Actor maximizes CVaR instead of expected Q-value
- **CVaR Computation**: Lines 359-363 compute Conditional Value at Risk from Q-distribution
- **Actor Loss**: `actor_loss = (log_probs * temperature - q_cvar).mean()` (Line 366)
- **Risk Parameter**: `cvar_risk` controls conservatism (0.1=risk-neutral, 0.9=very conservative)

### 3. Adaptive Action Limits
**Location**: `jaxrl5/agents/distributional_sac/distributional_agent.py:56-83`

**Implementation**: `TanhSquasher` class
```python
class TanhSquasher(nn.Module):
    def __call__(self, x: distrax.Distribution, output_range=None):
        high_scalar = self.param("high", lambda _: jnp.full((), self.init_value))
        high = high.at[..., self.high_idx].set(high_scalar)  # Learnable bound
        return TanhTransformedDistribution(x, low=low, high=high)
```

**Adaptive Limit Training** (Lines 381-430):
- **Learnable Action Bounds**: Lines 69-71 implement learnable upper bound for specific action dimension
- **CVaR-based Training**: Limits network trained using CVaR objective (Lines 412-416)
- **Separate Optimizer**: Uses dedicated learning rate `limits_lr: 1e-5` (Line 494)
- **Action Space Modification**: Dynamically adjusts action space bounds during training

### 4. Main Algorithm Integration
**Location**: `jaxrl5/agents/distributional_sac/distributional_agent.py:433-820`

**Class**: `DistributionalSACLearner`
- **Three-Component Training**: Lines 800-820 coordinate actor, critic, and limits updates
- **Distributional Bellman**: Lines 147-280 implement distributional Q-learning with ensemble reduction
- **Risk-Sensitive Policy**: Uses CVaR for both actor (Line 690) and limits (Line 741) training

**Configuration** (`distributional_limits_config.py`):
```python
config.cvar_risk = 0.9                    # Risk sensitivity level
config.num_atoms = 151                    # Distributional discretization
config.q_min = -100.0; config.q_max = 650.0  # Q-value range
config.limits_lr = 1e-5                   # Action limits learning rate
config.learned_action_space_idx = 1       # Which action dimension to learn limits for
```

### 5. Ensemble Uncertainty Quantification
**Location**: `jaxrl5/networks/ensemble.py:8-37`

**Implementation**:
- **Ensemble Q-Networks**: `num_qs` parallel Q-networks (default: 2) for epistemic uncertainty
- **Dropout Regularization**: Independent dropout for each ensemble member
- **Subsampling**: `subsample_ensemble()` for computational efficiency during target computation
- **Min/Mix Reduction**: Lines 230-246 show different ways to combine ensemble predictions

### 6. Safety-Critical Extensions
**Location**: `jaxrl5/agents/safety_critic_sac/safety_critic_sac.py`

**Additional Components**:
- **Safety Critic**: Lines 70-95 define separate network for constraint violation prediction
- **Safety Threshold**: Configurable safety constraint level
- **Safety Discount**: Different discount factor for safety vs. reward objectives

### Network Architecture
**Key Files**:
- `networks/mlp.py` - Multi-layer perceptron implementations
- `networks/ensemble.py` - Ensemble Q-networks for uncertainty estimation
- `networks/state_action_value.py` - Q-function architectures

## Simulation Environment

### Procedural World Generation
**File**: `offroad-procedural-sim/procedural_driving/procedural_arena.py`

**Features**:
- **Infinite Terrain**: Procedurally generated heightmaps using Perlin noise
- **Chunk System**: 3x3 grid of terrain chunks that update as vehicle moves
- **Configurable Worlds**: YAML-based terrain configuration (flat, bumpy, custom)

**World Configuration** (`configs/worlds/`):
- `flat.yaml` - Minimal terrain variation for basic testing
- `bumpy.yaml` - Challenging terrain with obstacles and elevation changes
- `default.yaml` - Balanced terrain for general training

### Vehicle Dynamics
**File**: `procedural_driving/car.py`

**Implementation**:
- MuJoCo-based physics simulation
- Realistic suspension, tire friction, and aerodynamics
- Sensor suite: IMU, GPS, camera (optional)

### Reward Function
**File**: `procedural_driving/task.py`

**Core Reward** (`batch_compute_reward`):
```python
# Reward = velocity component toward goal
velocities_to_goal = np.sum(car_vel_2d * directions_to_goal, axis=-1)
r = velocities_to_goal
```

**Additional Penalties**:
- Rollover detection (upside-down termination)
- Timeout penalties for navigation efficiency
- Optional: collision detection, terrain-based penalties

## Training Configuration

### Main Training Script
**File**: `scripts/sim/train_online_states.py`

**Working Training Command (MuJoCo/Procedural)**:
```bash
cd /home/risteon/workspace/gpudrive_docker/racer && \
source /home/risteon/miniconda3/bin/activate racer && \
DISPLAY=:0 WANDB_MODE=offline python scripts/sim/train_online_states.py \
  --config scripts/sim/configs/distributional_limits_config.py \
  --world_name flat
```

### Highway-Env Integration
**Training Command for Highway-Env**:
```bash
cd /home/risteon/workspace/gpudrive_docker/racer && \
source /home/risteon/miniconda3/bin/activate racer && \
DISPLAY=:0 WANDB_MODE=offline python scripts/sim/train_highway_states.py \
  --config scripts/sim/configs/highway_distributional_config.py \
  --max_steps 100000
```

**Key Differences from MuJoCo Training**:
- Uses `gymnasium` instead of `gym` for highway-env compatibility
- Implements collision risk safety reward based on inter-vehicle distances
- Flattened observation space (90D) from 15 vehicles × 6 features each
- Continuous action space (2D): [steering, acceleration]

**Highway-Env Setup**:
- **Environment**: `highway-v0` with continuous action space
- **Observation**: Kinematics data for 15 vehicles (presence, x, y, vx, vy, heading)
- **Safety Metric**: Collision risk based on minimum distance to other vehicles
- **Risk Parameters**: 
  - `collision_threshold = 2.0m` (imminent collision penalty)
  - `safety_threshold = 10.0m` (safe following distance)
- **Reward**: Base highway reward + collision risk penalty (exponential decay)

**Implementation Files**:
- **Main Script**: `scripts/sim/train_highway_states.py`
- **Configuration**: `scripts/sim/configs/highway_distributional_config.py`
- **Key Features**:
  - Collision risk safety reward function
  - Gymnasium compatibility wrapper
  - Bounded observation space for replay buffer compatibility
  - Speed and safety EMA tracking for logging

**Command Components**:
- `source /home/risteon/miniconda3/bin/activate racer` - Activate racer conda environment
- `DISPLAY=:0` - Set virtual display for headless MuJoCo rendering
- `WANDB_MODE=offline` - Run W&B in offline mode (no login required)
- `--config` - Algorithm configuration file (cvar_risk set in config, not CLI)
- `--world_name` - Terrain type: `flat`, `bumpy`, or `default`

**Alternative Usage** (if environment pre-activated):
```bash
DISPLAY=:0 WANDB_MODE=offline python scripts/sim/train_online_states.py \
  --config scripts/sim/configs/distributional_limits_config.py \
  --world_name flat
```

### Key Hyperparameters
**Common Settings**:
- `actor_lr: 3e-4` - Actor learning rate
- `critic_lr: 3e-4` - Critic learning rate  
- `discount: 0.99` - Reward discount factor
- `tau: 0.005` - Target network update rate
- `batch_size: 128` - Training batch size

**RACER-Specific**:
- `cvar_risk: 0.9` - Risk sensitivity level
- `num_atoms: 151` - Distributional discretization
- `limits_lr: 1e-5` - Action limit learning rate

### Evaluation Metrics
- Episode return and success rate
- CVaR risk metrics at different confidence levels
- Safety violations and constraint satisfaction
- Navigation efficiency (time to goal)

## Real Robot Setup

### Hardware Requirements
Based on FastRLAP setup:
- **Robot**: Jetson Orin NX for onboard compute
- **Workstation**: GPU-enabled training computer
- **Sensors**: GPS, IMU, cameras
- **Actuators**: VESC motor controller

### Deployment Workflow
1. **Goal Collection**: Record waypoints using `goal_graph_recorder_node`
2. **Training**: Run distributed training with sim-to-real data
3. **Inference**: Deploy policies using `real_inference.launch`

**Key ROS Nodes**:
- `inference_agent.py` - Policy execution
- `gps_state_estimator_node.py` - State estimation
- `data_collect.py` - Training data aggregation

## Development Workflows

### Training a New Policy
```bash
# 1. Configure algorithm
vim scripts/sim/configs/my_config.py

# 2. Start training (use working command with environment setup)
cd /home/risteon/workspace/gpudrive_docker/racer && \
source /home/risteon/miniconda3/bin/activate racer && \
DISPLAY=:0 WANDB_MODE=offline python scripts/sim/train_online_states.py \
  --config scripts/sim/configs/my_config.py \
  --world_name flat

# 3. Monitor with W&B
# Training metrics logged offline, sync with: wandb sync wandb/offline-run-*
```

### Testing Learned Policies
```bash
# Evaluate policy
python scripts/sim/run_policy.py --checkpoint_path ./checkpoints/

# Manual control for comparison
python scripts/sim/manual.py --world_name bumpy
```

### Sim-to-Real Transfer
```bash
# Robot side
roslaunch offroad_bringup real_inference.launch

# Training side  
python offroad_learning/src/offroad_learning/training/training.py
```

## Key Algorithmic Insights

### RACER Architecture Overview
**Three-Component System**: RACER integrates three key components for safe autonomous driving:

1. **Epistemic vs Aleatoric Uncertainty**: 
   - **Ensemble**: Provides epistemic uncertainty (model uncertainty)
   - **Distributional**: Captures aleatoric uncertainty (environment stochasticity)

2. **Risk-Sensitive Control**: 
   - **CVaR Objective**: Focuses on worst-case scenarios rather than expected performance
   - **Conservative Policy**: `cvar_risk=0.9` trains agent to avoid tail risks

3. **Adaptive Safety Constraints**: 
   - **Learned Action Limits**: Discovers safe operating regions during training
   - **Dynamic Bounds**: Action space adapts based on learned risk assessment

### Implementation Hierarchy
```
DistributionalSACLearner (Main Algorithm)
├── DistributionalStateActionValue (Epistemic Critic)
│   ├── Ensemble (Uncertainty Quantification)
│   └── Categorical Distribution (Aleatoric Uncertainty)
├── CVaR Actor (Risk-Sensitive Policy)
│   └── cvar() function (Tail Risk Optimization)
└── TanhSquasher (Adaptive Action Limits)
    └── Learnable Action Bounds
```

### Training Flow
1. **Critic Update**: Distributional Bellman backup with ensemble reduction
2. **Actor Update**: CVaR-based policy gradient (risk-sensitive)
3. **Limits Update**: CVaR-based action bound learning (adaptive safety)

## Key Files Reference

### Algorithm Core
- `jaxrl5/agents/distributional_sac/distributional_agent.py:316-324` - CVaR risk measure implementation
- `jaxrl5/agents/distributional_sac/distributional_agent.py:56-83` - Adaptive action limits (TanhSquasher)
- `jaxrl5/agents/distributional_sac/distributional_agent.py:33-55` - Distributional Q-networks
- `jaxrl5/agents/distributional_sac/distributional_agent.py:433-820` - Main RACER learner class
- `jaxrl5/networks/ensemble.py:8-37` - Ensemble uncertainty quantification
- `jaxrl5/agents/safety_critic_sac/safety_critic_sac.py` - Safety-critical extensions

### Environment  
- `procedural_driving/procedural_arena.py` - World generation
- `procedural_driving/task.py` - Reward functions and episode logic
- `procedural_driving/car.py` - Vehicle dynamics

### Training
- `scripts/sim/train_online_states.py` - Main training loop
- `scripts/sim/configs/distributional_limits_config.py` - RACER hyperparameters
- `jaxrl5/data/replay_buffer.py` - Experience replay

### Real Robot
- `offroad_learning/src/offroad_learning/inference/inference_agent.py` - Policy deployment
- `offroad_learning/src/offroad_learning/training/data_collect.py` - Data collection
- `offroad_sensor_fusion/src/offroad_sensor_fusion/gps_state_estimator_node.py` - State estimation

## Installation and Setup

```bash
# Create environment
conda create -n racer python=3.11
conda activate racer

# Install dependencies
pip install -e jaxrl5[train] dmcgym .
pip install -r requirements.txt

# For real robot (additional)
cd offroad-robot-ros
pip install -r requirements.txt
```

## Research Applications

**Primary Use Cases**:
- Safe autonomous navigation in unstructured environments
- Risk-sensitive policy learning for safety-critical systems
- Sim-to-real transfer for offroad robotics
- Distributional RL algorithm development

**Experimental Scenarios**:
- Varying terrain difficulty (flat → bumpy)
- Different risk sensitivity levels (cvar_risk: 0.1 → 0.9)
- Comparison with standard SAC and safety-critical baselines
- Real-world validation on physical robots

This implementation provides a complete pipeline from algorithm development through real-world deployment for risk-sensitive autonomous driving research.