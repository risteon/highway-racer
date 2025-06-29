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

## Key Files
- `scripts/sim/train_highway_states.py` - Main highway training script
- `scripts/sim/configs/highway_distributional_config.py` - Highway-specific RACER configuration
- Uses `highway-v0` environment with continuous action control and collision risk safety rewards