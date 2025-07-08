#!/bin/bash

# analyze_q_values.sh - Complete Q-value analysis pipeline for RACER
# 
# This script automates the extraction and visualization of Q-value distributions
# from trained RACER (Distributional SAC) policies on highway environments.
#
# Usage: ./analyze_q_values.sh <checkpoint_path> <config_path> [num_episodes] [max_steps]
#
# Arguments:
#   checkpoint_path : Path to policy checkpoint directory (required)
#   config_path     : Path to training config file (required)  
#   num_episodes    : Number of episodes to analyze (optional, default: 3)
#   max_steps       : Maximum steps per episode (optional, default: 300)
#
# Example:
#   ./analyze_q_values.sh policies/glorious-sea-116/checkpoint_500000 scripts/sim/configs/hpd_collision.py
#   ./analyze_q_values.sh policies/run-42/checkpoint_100000 configs/my_config.py 5 500
#
# Output Structure:
#   ./q_values/<run_name>/<checkpoint_name>/
#   ├── data/                           # Q-value data and videos
#   │   ├── episode_0/
#   │   │   ├── q_values_and_trajectory.npz
#   │   │   ├── episode_0_video.mp4
#   │   │   └── episode_summary.json
#   │   └── aggregate_analysis.json
#   └── plots/                          # Generated visualizations
#       ├── q_value_evolution.png
#       ├── q_distributions_selected_steps.png
#       └── action_preference_analysis.png

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%H:%M:%S') $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%H:%M:%S') $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $1"
}

# Help function
show_help() {
    echo "Q-Value Analysis Pipeline for RACER"
    echo ""
    echo "Usage: $0 <checkpoint_path> <config_path> [num_episodes] [max_steps]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_path   Path to policy checkpoint directory (required)"
    echo "  config_path       Path to training config file (required)"
    echo "  num_episodes      Number of episodes to analyze (default: 3)"
    echo "  max_steps         Maximum steps per episode (default: 300)"
    echo ""
    echo "Examples:"
    echo "  $0 policies/glorious-sea-116/checkpoint_500000 scripts/sim/configs/hpd_collision.py"
    echo "  $0 policies/run-42/checkpoint_100000 configs/my_config.py 5 500"
    echo ""
    echo "Output will be saved to: ./q_values/<run_name>/<checkpoint_name>/"
}

# Check if help requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Validate arguments
if [ $# -lt 2 ]; then
    log_error "Missing required arguments"
    show_help
    exit 1
fi

CHECKPOINT_PATH="$1"
CONFIG_PATH="$2"
NUM_EPISODES="${3:-3}"  # Default to 3 episodes
MAX_STEPS="${4:-300}"   # Default to 300 steps

log_info "Starting Q-value analysis pipeline"
log_info "Checkpoint: $CHECKPOINT_PATH"
log_info "Config: $CONFIG_PATH"
log_info "Episodes: $NUM_EPISODES, Max steps: $MAX_STEPS"

# Validate checkpoint path
if [ ! -d "$CHECKPOINT_PATH" ]; then
    log_error "Checkpoint directory does not exist: $CHECKPOINT_PATH"
    exit 1
fi

# Validate config file
if [ ! -f "$CONFIG_PATH" ]; then
    log_error "Config file does not exist: $CONFIG_PATH"
    exit 1
fi

# Extract run name and checkpoint name from path
CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH")
RUN_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")

log_info "Detected run: $RUN_NAME, checkpoint: $CHECKPOINT_NAME"

# Create output directory structure
OUTPUT_BASE="./q_values"
OUTPUT_DIR="$OUTPUT_BASE/$RUN_NAME/$CHECKPOINT_NAME"
DATA_DIR="$OUTPUT_DIR/data"
PLOTS_DIR="$OUTPUT_DIR/plots"

log_info "Creating output directories..."
mkdir -p "$DATA_DIR"
mkdir -p "$PLOTS_DIR"

# Extract CVaR risk from config file for visualization
CVAR_RISK="0.8"  # Default value
if [ -f "$CONFIG_PATH" ]; then
    # Try to extract cvar_risk from config file
    EXTRACTED_CVAR=$(grep -o "config\.cvar_risk = [0-9.]*" "$CONFIG_PATH" | tail -1 | cut -d'=' -f2 | tr -d ' ')
    if [ ! -z "$EXTRACTED_CVAR" ]; then
        CVAR_RISK="$EXTRACTED_CVAR"
        log_info "Extracted CVaR risk from config: $CVAR_RISK"
    else
        log_warning "Could not extract CVaR risk from config, using default: $CVAR_RISK"
    fi
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    log_error "Conda not found. Please install conda or ensure it's in your PATH"
    exit 1
fi

# Check if racer environment exists
if ! conda env list | grep -q "^racer "; then
    log_error "Conda environment 'racer' not found. Please create it first"
    exit 1
fi

log_info "Activating conda environment 'racer'..."
# Source conda initialization
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    log_error "Could not find conda initialization script"
    exit 1
fi

conda activate racer

# Verify required scripts exist
RECORD_SCRIPT="scripts/sim/record_q_values.py"
PLOT_SCRIPT="scripts/sim/plot_q_values.py"

if [ ! -f "$RECORD_SCRIPT" ]; then
    log_error "Q-value recording script not found: $RECORD_SCRIPT"
    exit 1
fi

if [ ! -f "$PLOT_SCRIPT" ]; then
    log_error "Q-value plotting script not found: $PLOT_SCRIPT"
    exit 1
fi

log_success "Environment setup complete"

# Phase 1: Q-value recording
log_info "Phase 1: Recording Q-value distributions..."
log_info "This will analyze $NUM_EPISODES episodes with up to $MAX_STEPS steps each"

WANDB_MODE=disabled python "$RECORD_SCRIPT" \
    --policy_file "$CHECKPOINT_PATH" \
    --config "$CONFIG_PATH" \
    --num_episodes "$NUM_EPISODES" \
    --max_steps "$MAX_STEPS" \
    --output_dir "$DATA_DIR" \
    --render=true

if [ $? -eq 0 ]; then
    log_success "Q-value recording completed successfully"
else
    log_error "Q-value recording failed"
    exit 1
fi

# Find the actual data directory (script creates run_name/checkpoint_name subdirs)
ACTUAL_DATA_DIR=$(find "$DATA_DIR" -name "episode_0" -type d | head -1 | dirname)

if [ -z "$ACTUAL_DATA_DIR" ]; then
    log_error "Could not find recorded episode data"
    exit 1
fi

log_info "Q-value data saved to: $ACTUAL_DATA_DIR"

# Phase 2: Visualization
log_info "Phase 2: Generating visualizations..."

# Run visualization for each episode
EPISODE_DIRS=$(find "$ACTUAL_DATA_DIR" -name "episode_*" -type d | sort)
EPISODE_COUNT=$(echo "$EPISODE_DIRS" | wc -l)

log_info "Found $EPISODE_COUNT episodes to visualize"

# Create visualizations for the first episode (can be extended for multiple)
FIRST_EPISODE=$(echo "$EPISODE_DIRS" | head -1)

if [ ! -z "$FIRST_EPISODE" ]; then
    log_info "Creating visualizations for: $(basename "$FIRST_EPISODE")"
    
    python "$PLOT_SCRIPT" \
        --data_dir "$FIRST_EPISODE" \
        --output_dir "$PLOTS_DIR" \
        --cvar_risk "$CVAR_RISK" \
        --create_animation=false \
        --save_static_plots=true
    
    if [ $? -eq 0 ]; then
        log_success "Visualization completed successfully"
    else
        log_warning "Visualization completed with warnings (likely animation issues)"
    fi
else
    log_error "No episode data found for visualization"
    exit 1
fi

# Generate summary report
log_info "Generating summary report..."

SUMMARY_FILE="$OUTPUT_DIR/analysis_summary.txt"

cat > "$SUMMARY_FILE" << EOF
Q-Value Analysis Summary
========================

Analysis Date: $(date)
Checkpoint: $CHECKPOINT_PATH
Config: $CONFIG_PATH
Run Name: $RUN_NAME
Checkpoint: $CHECKPOINT_NAME

Parameters:
- Episodes Analyzed: $NUM_EPISODES
- Max Steps per Episode: $MAX_STEPS
- CVaR Risk Level: $CVAR_RISK

Output Structure:
$OUTPUT_DIR/
├── data/                    # Q-value data and videos
│   ├── episode_*/
│   │   ├── q_values_and_trajectory.npz
│   │   ├── episode_*_video.mp4
│   │   └── episode_summary.json
│   └── aggregate_analysis.json
└── plots/                   # Generated visualizations
    ├── q_value_evolution.png
    ├── q_distributions_selected_steps.png
    └── action_preference_analysis.png

Files Generated:
EOF

# List all generated files
find "$OUTPUT_DIR" -type f | sort >> "$SUMMARY_FILE"

# Extract some basic statistics if possible
if command -v python &> /dev/null; then
    python << EOF >> "$SUMMARY_FILE"

# Extract episode statistics
import numpy as np
import os
import glob

print("\nEpisode Statistics:")
print("==================")

episode_dirs = sorted(glob.glob("$ACTUAL_DATA_DIR/episode_*"))
for episode_dir in episode_dirs:
    npz_file = os.path.join(episode_dir, "q_values_and_trajectory.npz")
    if os.path.exists(npz_file):
        try:
            data = np.load(npz_file)
            episode_name = os.path.basename(episode_dir)
            print(f"{episode_name}:")
            print(f"  Return: {data['episode_return']:.2f}")
            print(f"  Length: {data['episode_length']} steps")
            print(f"  Frames: {len(data['frames'])}")
            
            # Q-value statistics
            q_accel_mean = (data['q_probs_accelerate'] * data['q_atoms_accelerate']).sum(axis=-1).mean()
            q_brake_mean = (data['q_probs_brake'] * data['q_atoms_brake']).sum(axis=-1).mean()
            print(f"  Mean Q-value (accelerate): {q_accel_mean:.2f}")
            print(f"  Mean Q-value (brake): {q_brake_mean:.2f}")
            print(f"  Q-value preference: {'Accelerate' if q_accel_mean > q_brake_mean else 'Brake'}")
            print()
        except Exception as e:
            print(f"  Error reading {episode_name}: {e}")
EOF
fi

log_success "Analysis complete!"
echo ""
echo "=========================="
echo "    ANALYSIS SUMMARY      "
echo "=========================="
echo "📁 Output Directory: $OUTPUT_DIR"
echo "📊 Episodes Analyzed: $NUM_EPISODES"
echo "🎯 CVaR Risk Level: $CVAR_RISK"
echo "📈 Plots Generated: $(find "$PLOTS_DIR" -name "*.png" | wc -l)"
echo "🎥 Videos Generated: $(find "$ACTUAL_DATA_DIR" -name "*.mp4" | wc -l)"
echo "📄 Summary Report: $SUMMARY_FILE"
echo ""
echo "Generated Files:"
echo "  • Q-value evolution plots"
echo "  • Distribution snapshots"  
echo "  • Action preference analysis"
echo "  • Episode videos with Q-value data"
echo ""
echo "To view plots: ls $PLOTS_DIR"
echo "To view data: ls $ACTUAL_DATA_DIR"
echo ""
log_success "Q-value analysis pipeline completed successfully! 🚀"