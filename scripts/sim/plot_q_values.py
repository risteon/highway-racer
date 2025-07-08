#! /usr/bin/env python
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
from pathlib import Path
from absl import app, flags
from tqdm import tqdm
import argparse

# Set up plotting style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_palette("husl")

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None, "Path to the Q-value data directory (episode folder)")
flags.DEFINE_string("output_dir", "./q_value_plots", "Path to save visualization outputs")
flags.DEFINE_boolean("create_animation", True, "Create animated GIF of Q-values over time")
flags.DEFINE_boolean("save_static_plots", True, "Save static analysis plots")
flags.DEFINE_integer("plot_every_n_steps", 5, "Plot every N steps for static analysis")
flags.DEFINE_float("cvar_risk", 0.9, "CVaR risk level for risk-sensitive analysis")
flags.DEFINE_boolean("show_ensemble", True, "Show individual ensemble members")


def load_episode_data(data_dir):
    """
    Load episode data from NPZ file.
    
    Args:
        data_dir: Path to episode directory containing q_values_and_trajectory.npz
    
    Returns:
        data: Dict with loaded arrays
        metadata: Episode metadata from JSON
    """
    npz_file = os.path.join(data_dir, "q_values_and_trajectory.npz")
    summary_file = os.path.join(data_dir, "episode_summary.json")
    
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")
    
    # Load main data
    data = dict(np.load(npz_file))
    
    # Load metadata if available
    metadata = {}
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            metadata = json.load(f)
    
    print(f"Loaded episode data from: {data_dir}")
    print(f"Episode length: {data['episode_length']}")
    print(f"Episode return: {data['episode_return']:.2f}")
    print(f"Number of frames: {len(data['frames'])}")
    print(f"Q-value shapes:")
    print(f"  Speed Control:")
    print(f"    Accelerate: {data['q_probs_accelerate'].shape}")
    print(f"    Brake: {data['q_probs_brake'].shape}")
    print(f"    Continue: {data['q_probs_continue'].shape}")
    print(f"  Steering:")
    print(f"    Steer Right: {data['q_probs_steer_right'].shape}")
    print(f"    Steer Left: {data['q_probs_steer_left'].shape}")
    print(f"  Policy: {data['q_probs_policy'].shape}")
    
    return data, metadata


def compute_q_statistics(q_probs, q_atoms, cvar_risk=0.9):
    """
    Compute Q-value statistics from distributions.
    
    Args:
        q_probs: Q-value probabilities (T, N_ensemble, Batch, N_atoms)
        q_atoms: Q-value atoms (T, N_ensemble, Batch, N_atoms)
        cvar_risk: Risk level for CVaR computation
    
    Returns:
        stats: Dict with computed statistics
    """
    # Squeeze out the batch dimension if it exists
    if q_probs.ndim == 4 and q_probs.shape[2] == 1:
        q_probs = q_probs.squeeze(axis=2)  # (T, N_ensemble, N_atoms)
        q_atoms = q_atoms.squeeze(axis=2)  # (T, N_ensemble, N_atoms)
    
    # Compute expected Q-values (mean of distribution)
    q_mean = np.sum(q_probs * q_atoms, axis=-1)  # (T, N_ensemble)
    
    # Compute Q-value variance (uncertainty)
    q_variance = np.sum(q_probs * (q_atoms - q_mean[..., None])**2, axis=-1)
    q_std = np.sqrt(q_variance)
    
    # Compute CVaR (Conditional Value at Risk)
    def compute_cvar(probs, atoms, risk):
        # Sort atoms and corresponding probabilities
        sorted_indices = np.argsort(atoms, axis=-1)
        sorted_atoms = np.take_along_axis(atoms, sorted_indices, axis=-1)
        sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
        
        # Compute CDF
        cdf = np.cumsum(sorted_probs, axis=-1)
        
        # Find CVaR threshold
        risk_mask = cdf <= risk
        if risk == 0.0:
            # Risk-neutral case: return expected value
            return np.sum(sorted_probs * sorted_atoms, axis=-1)
        
        # Compute CVaR
        cvar_probs = np.where(risk_mask, sorted_probs / risk, 0.0)
        cvar = np.sum(cvar_probs * sorted_atoms, axis=-1)
        
        return cvar
    
    q_cvar = compute_cvar(q_probs, q_atoms, cvar_risk)
    
    stats = {
        'mean': q_mean,           # Expected Q-value
        'std': q_std,             # Q-value uncertainty  
        'variance': q_variance,   # Q-value variance
        'cvar': q_cvar,           # Conditional Value at Risk
        'ensemble_mean': np.mean(q_mean, axis=-1),      # Mean across ensemble
        'ensemble_std': np.std(q_mean, axis=-1),        # Epistemic uncertainty
        'aleatoric_mean': np.mean(q_std, axis=-1),      # Aleatoric uncertainty
    }
    
    return stats


def create_static_analysis_plots(data, output_dir, plot_every_n=5, cvar_risk=0.9):
    """
    Create comprehensive static analysis plots.
    """
    episode_length = int(data['episode_length'])
    steps_to_plot = range(0, episode_length, plot_every_n)
    
    # Compute statistics for all actions
    actions = ['accelerate', 'brake', 'continue', 'steer_right', 'steer_left', 'policy']
    stats = {}
    
    for action in actions:
        stats[action] = compute_q_statistics(
            data[f'q_probs_{action}'], 
            data[f'q_atoms_{action}'], 
            cvar_risk
        )
    
    # 1. Q-Value Evolution Over Time (Extended Layout)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Colors for each action
    colors = {
        'accelerate': 'blue',
        'brake': 'orange', 
        'continue': 'green',
        'steer_right': 'red',
        'steer_left': 'purple',
        'policy': 'brown'
    }
    
    # Expected Q-values - Speed Control Actions
    axes[0, 0].plot(stats['accelerate']['ensemble_mean'], label='Accelerate', color=colors['accelerate'], linewidth=2)
    axes[0, 0].plot(stats['brake']['ensemble_mean'], label='Brake', color=colors['brake'], linewidth=2)
    axes[0, 0].plot(stats['continue']['ensemble_mean'], label='Continue', color=colors['continue'], linewidth=2)
    axes[0, 0].fill_between(range(len(stats['accelerate']['ensemble_mean'])), 
                           stats['accelerate']['ensemble_mean'] - stats['accelerate']['ensemble_std'],
                           stats['accelerate']['ensemble_mean'] + stats['accelerate']['ensemble_std'], 
                           alpha=0.2, color=colors['accelerate'])
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Expected Q-Value')
    axes[0, 0].set_title('Speed Control Q-Values')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Expected Q-values - Steering Actions  
    axes[0, 1].plot(stats['steer_right']['ensemble_mean'], label='Steer Right', color=colors['steer_right'], linewidth=2)
    axes[0, 1].plot(stats['steer_left']['ensemble_mean'], label='Steer Left', color=colors['steer_left'], linewidth=2)
    axes[0, 1].plot(stats['continue']['ensemble_mean'], label='Continue', color=colors['continue'], linewidth=2)
    axes[0, 1].fill_between(range(len(stats['steer_right']['ensemble_mean'])), 
                           stats['steer_right']['ensemble_mean'] - stats['steer_right']['ensemble_std'],
                           stats['steer_right']['ensemble_mean'] + stats['steer_right']['ensemble_std'], 
                           alpha=0.2, color=colors['steer_right'])
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Expected Q-Value')
    axes[0, 1].set_title('Steering Control Q-Values')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Policy vs Best Actions
    axes[0, 2].plot(stats['policy']['ensemble_mean'], label='Policy Action', color=colors['policy'], linewidth=3)
    # Find best action at each step
    best_q_values = []
    for step in range(len(stats['policy']['ensemble_mean'])):
        step_q_values = {action: stats[action]['ensemble_mean'][step] for action in actions[:-1]}  # Exclude policy
        best_action = max(step_q_values, key=step_q_values.get)
        best_q_values.append(step_q_values[best_action])
    axes[0, 2].plot(best_q_values, label='Best Available Action', color='black', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('Time Step')
    axes[0, 2].set_ylabel('Expected Q-Value')
    axes[0, 2].set_title('Policy vs Optimal Q-Values')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # CVaR Q-values (Risk-Sensitive)
    axes[1, 0].plot(np.mean(stats['accelerate']['cvar'], axis=-1), label=f'Accelerate', color=colors['accelerate'], linewidth=2)
    axes[1, 0].plot(np.mean(stats['brake']['cvar'], axis=-1), label=f'Brake', color=colors['brake'], linewidth=2)
    axes[1, 0].plot(np.mean(stats['continue']['cvar'], axis=-1), label=f'Continue', color=colors['continue'], linewidth=2)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('CVaR Q-Value')
    axes[1, 0].set_title(f'Speed Control CVaR (α={cvar_risk})')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Steering CVaR
    axes[1, 1].plot(np.mean(stats['steer_right']['cvar'], axis=-1), label=f'Steer Right', color=colors['steer_right'], linewidth=2)
    axes[1, 1].plot(np.mean(stats['steer_left']['cvar'], axis=-1), label=f'Steer Left', color=colors['steer_left'], linewidth=2)
    axes[1, 1].plot(np.mean(stats['continue']['cvar'], axis=-1), label=f'Continue', color=colors['continue'], linewidth=2)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('CVaR Q-Value')
    axes[1, 1].set_title(f'Steering Control CVaR (α={cvar_risk})')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Overall Uncertainty Analysis
    for action in ['accelerate', 'brake', 'steer_right', 'steer_left']:
        axes[1, 2].plot(stats[action]['ensemble_std'], label=f'{action.replace("_", " ").title()}', 
                       color=colors[action], linewidth=2)
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Ensemble Std Dev')
    axes[1, 2].set_title('Epistemic Uncertainty by Action')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'q_value_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Q-Distribution Heatmaps for Selected Steps (Extended for 6 actions)
    n_plots = min(4, len(steps_to_plot))  # Show 4 time steps
    fig, axes = plt.subplots(6, n_plots, figsize=(4*n_plots, 20))
    if n_plots == 1:
        axes = axes.reshape(-1, 1)
    
    # Action order for display
    action_names = ['accelerate', 'brake', 'continue', 'steer_right', 'steer_left', 'policy']
    action_titles = ['Accelerate', 'Brake', 'Continue', 'Steer Right', 'Steer Left', 'Policy Action']
    
    for i, step in enumerate(list(steps_to_plot)[:n_plots]):
        for j, (action, title) in enumerate(zip(action_names, action_titles)):
            # Get Q-distributions at this step (average over ensemble and squeeze batch dim)
            q_probs = np.mean(data[f'q_probs_{action}'][step].squeeze(), axis=0)
            q_atoms = np.mean(data[f'q_atoms_{action}'][step].squeeze(), axis=0)
            
            # Calculate bar width safely
            if len(q_atoms) > 1:
                width = (q_atoms[1] - q_atoms[0]) * 0.8
            else:
                width = 1.0
            
            # Plot distribution with action-specific color
            axes[j, i].bar(q_atoms, q_probs, width=width, alpha=0.7, color=colors[action])
            axes[j, i].set_title(f'{title} (Step {step})')
            if j == len(action_names) - 1:  # Bottom row
                axes[j, i].set_xlabel('Q-Value')
            axes[j, i].set_ylabel('Probability')
            axes[j, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'q_distributions_selected_steps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Enhanced Action Preference Analysis
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Speed Control Preferences (top row)
    q_diff_accel_brake = stats['accelerate']['ensemble_mean'] - stats['brake']['ensemble_mean']
    q_diff_accel_continue = stats['accelerate']['ensemble_mean'] - stats['continue']['ensemble_mean']
    q_diff_brake_continue = stats['brake']['ensemble_mean'] - stats['continue']['ensemble_mean']
    
    axes[0, 0].plot(q_diff_accel_brake, label='Accelerate - Brake', color='blue', linewidth=2)
    axes[0, 0].plot(q_diff_accel_continue, label='Accelerate - Continue', color='green', linewidth=2)
    axes[0, 0].plot(q_diff_brake_continue, label='Brake - Continue', color='orange', linewidth=2)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].fill_between(range(len(q_diff_accel_brake)), 0, q_diff_accel_brake, 
                           where=(q_diff_accel_brake > 0), alpha=0.2, color='blue')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Q-Value Difference')
    axes[0, 0].set_title('Speed Control Preferences')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Steering Control Preferences
    q_diff_right_left = stats['steer_right']['ensemble_mean'] - stats['steer_left']['ensemble_mean']
    q_diff_right_continue = stats['steer_right']['ensemble_mean'] - stats['continue']['ensemble_mean']
    q_diff_left_continue = stats['steer_left']['ensemble_mean'] - stats['continue']['ensemble_mean']
    
    axes[0, 1].plot(q_diff_right_left, label='Steer Right - Steer Left', color='red', linewidth=2)
    axes[0, 1].plot(q_diff_right_continue, label='Steer Right - Continue', color='purple', linewidth=2)
    axes[0, 1].plot(q_diff_left_continue, label='Steer Left - Continue', color='brown', linewidth=2)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].fill_between(range(len(q_diff_right_left)), 0, q_diff_right_left, 
                           where=(q_diff_right_left > 0), alpha=0.2, color='red', label='Prefer Right')
    axes[0, 1].fill_between(range(len(q_diff_right_left)), 0, q_diff_right_left, 
                           where=(q_diff_right_left < 0), alpha=0.2, color='purple', label='Prefer Left')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Q-Value Difference')
    axes[0, 1].set_title('Steering Control Preferences')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Actual actions taken (middle row)
    actions_taken = data['actions_taken']
    steering_actions = actions_taken[:, 0]  # First dimension is steering
    acceleration_actions = actions_taken[:, 1]  # Second dimension is acceleration
    
    axes[1, 0].plot(acceleration_actions, label='Acceleration Actions', linewidth=2, color='blue')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=1, color='green', linestyle=':', alpha=0.7, label='Max Accelerate')
    axes[1, 0].axhline(y=-1, color='red', linestyle=':', alpha=0.7, label='Max Brake')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Acceleration Action')
    axes[1, 0].set_title('Actual Acceleration Actions')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(steering_actions, label='Steering Actions', linewidth=2, color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=1, color='green', linestyle=':', alpha=0.7, label='Max Right')
    axes[1, 1].axhline(y=-1, color='purple', linestyle=':', alpha=0.7, label='Max Left')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Steering Action')
    axes[1, 1].set_title('Actual Steering Actions')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Action preference heatmap (bottom row)
    action_preferences = np.zeros((len(actions) - 1, episode_length))  # Exclude policy
    for step in range(episode_length):
        step_q_values = []
        for action in actions[:-1]:  # Exclude policy
            step_q_values.append(stats[action]['ensemble_mean'][step])
        
        # Normalize to get preference scores (softmax-like)
        step_q_values = np.array(step_q_values)
        exp_values = np.exp(step_q_values - np.max(step_q_values))
        preferences = exp_values / np.sum(exp_values)
        action_preferences[:, step] = preferences
    
    im1 = axes[2, 0].imshow(action_preferences, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    axes[2, 0].set_xlabel('Time Step')
    axes[2, 0].set_ylabel('Action')
    axes[2, 0].set_title('Action Preference Heatmap')
    axes[2, 0].set_yticks(range(len(actions) - 1))
    axes[2, 0].set_yticklabels([a.replace('_', ' ').title() for a in actions[:-1]])
    plt.colorbar(im1, ax=axes[2, 0], label='Preference Score')
    
    # CVaR vs Expected Q-value comparison
    for action in ['accelerate', 'brake', 'steer_right', 'steer_left']:
        expected_q = stats[action]['ensemble_mean']
        cvar_q = np.mean(stats[action]['cvar'], axis=-1)
        risk_sensitivity = expected_q - cvar_q
        axes[2, 1].plot(risk_sensitivity, label=f'{action.replace("_", " ").title()}', 
                       color=colors[action], linewidth=2)
    
    axes[2, 1].set_xlabel('Time Step')
    axes[2, 1].set_ylabel('Expected Q - CVaR Q')
    axes[2, 1].set_title('Risk Sensitivity by Action')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_preference_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Static analysis plots saved to: {output_dir}")


def create_animated_visualization(data, output_dir, fps=5):
    """
    Create animated visualization showing Q-distributions evolving with environment frames.
    """
    episode_length = int(data['episode_length'])
    frames = data['frames']
    
    if len(frames) == 0:
        print("No frames available for animation")
        return
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Environment frame (top left, spanning 2x2)
    ax_frame = fig.add_subplot(gs[0:2, 0:2])
    
    # Q-distribution plots (right side)
    ax_accel = fig.add_subplot(gs[0, 2])
    ax_brake = fig.add_subplot(gs[0, 3])
    ax_policy = fig.add_subplot(gs[1, 2])
    ax_comparison = fig.add_subplot(gs[1, 3])
    
    # Time series plots (bottom, spanning full width)
    ax_evolution = fig.add_subplot(gs[2, :])
    
    # Initialize plots
    frame_im = ax_frame.imshow(frames[0])
    ax_frame.set_title('Environment')
    ax_frame.axis('off')
    
    # Get Q-value ranges for consistent axes
    all_atoms = np.concatenate([
        data['q_atoms_accelerate'].flatten(),
        data['q_atoms_brake'].flatten(),
        data['q_atoms_policy'].flatten()
    ])
    q_min, q_max = np.min(all_atoms), np.max(all_atoms)
    prob_max = np.max([
        data['q_probs_accelerate'].max(),
        data['q_probs_brake'].max(),
        data['q_probs_policy'].max()
    ])
    
    def animate(step):
        # Clear previous plots
        ax_accel.clear()
        ax_brake.clear()
        ax_policy.clear()
        ax_comparison.clear()
        ax_evolution.clear()
        
        # Update environment frame
        if step < len(frames):
            frame_im.set_array(frames[step])
            ax_frame.set_title(f'Environment (Step {step})')
        
        # Get Q-distributions at current step (average over ensemble)
        q_probs_accel = np.mean(data['q_probs_accelerate'][step], axis=0)
        q_atoms_accel = np.mean(data['q_atoms_accelerate'][step], axis=0)
        q_probs_brake = np.mean(data['q_probs_brake'][step], axis=0)
        q_atoms_brake = np.mean(data['q_atoms_brake'][step], axis=0)
        q_probs_policy = np.mean(data['q_probs_policy'][step], axis=0)
        q_atoms_policy = np.mean(data['q_atoms_policy'][step], axis=0)
        
        # Plot Q-distributions
        width = (q_atoms_accel[1] - q_atoms_accel[0]) * 0.8
        
        ax_accel.bar(q_atoms_accel, q_probs_accel, width=width, alpha=0.7, color='blue')
        ax_accel.set_title('Accelerate')
        ax_accel.set_xlim(q_min, q_max)
        ax_accel.set_ylim(0, prob_max * 1.1)
        ax_accel.set_xlabel('Q-Value')
        ax_accel.set_ylabel('Probability')
        
        ax_brake.bar(q_atoms_brake, q_probs_brake, width=width, alpha=0.7, color='orange')
        ax_brake.set_title('Brake')
        ax_brake.set_xlim(q_min, q_max)
        ax_brake.set_ylim(0, prob_max * 1.1)
        ax_brake.set_xlabel('Q-Value')
        ax_brake.set_ylabel('Probability')
        
        ax_policy.bar(q_atoms_policy, q_probs_policy, width=width, alpha=0.7, color='green')
        ax_policy.set_title('Policy Action')
        ax_policy.set_xlim(q_min, q_max)
        ax_policy.set_ylim(0, prob_max * 1.1)
        ax_policy.set_xlabel('Q-Value')
        ax_policy.set_ylabel('Probability')
        
        # Comparison plot (overlaid distributions)
        ax_comparison.bar(q_atoms_accel, q_probs_accel, width=width*0.8, alpha=0.5, 
                         color='blue', label='Accelerate')
        ax_comparison.bar(q_atoms_brake, q_probs_brake, width=width*0.8, alpha=0.5, 
                         color='orange', label='Brake')
        ax_comparison.set_title('Comparison')
        ax_comparison.set_xlim(q_min, q_max)
        ax_comparison.set_ylim(0, prob_max * 1.1)
        ax_comparison.set_xlabel('Q-Value')
        ax_comparison.set_ylabel('Probability')
        ax_comparison.legend()
        
        # Evolution plot (time series up to current step)
        steps_so_far = range(step + 1)
        
        # Compute statistics up to current step
        stats_accel = compute_q_statistics(
            data['q_probs_accelerate'][:step+1], 
            data['q_atoms_accelerate'][:step+1]
        )
        stats_brake = compute_q_statistics(
            data['q_probs_brake'][:step+1], 
            data['q_atoms_brake'][:step+1]
        )
        
        ax_evolution.plot(steps_so_far, stats_accel['ensemble_mean'], 
                         label='Accelerate', linewidth=2, color='blue')
        ax_evolution.plot(steps_so_far, stats_brake['ensemble_mean'], 
                         label='Brake', linewidth=2, color='orange')
        
        # Highlight current step
        if step > 0:
            ax_evolution.scatter([step], [stats_accel['ensemble_mean'][-1]], 
                               color='blue', s=50, zorder=5)
            ax_evolution.scatter([step], [stats_brake['ensemble_mean'][-1]], 
                               color='orange', s=50, zorder=5)
        
        ax_evolution.set_xlabel('Time Step')
        ax_evolution.set_ylabel('Expected Q-Value')
        ax_evolution.set_title('Q-Value Evolution Over Time')
        ax_evolution.legend()
        ax_evolution.grid(True)
        ax_evolution.set_xlim(0, episode_length)
        
        return [frame_im]
    
    # Create animation
    print(f"Creating animation with {episode_length} frames...")
    anim = FuncAnimation(fig, animate, frames=episode_length, interval=1000//fps, blit=False)
    
    # Save as GIF
    gif_path = os.path.join(output_dir, 'q_value_evolution_animated.gif')
    print("Saving animation (this may take a while)...")
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close()
    
    print(f"Animation saved to: {gif_path}")


def main(_):
    if FLAGS.data_dir is None:
        print("Error: --data_dir must be specified")
        print("Usage: python plot_q_values.py --data_dir=/path/to/episode_directory")
        return
    
    # Create output directory
    Path(FLAGS.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load episode data
    try:
        data, metadata = load_episode_data(FLAGS.data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the data directory contains 'q_values_and_trajectory.npz'")
        return
    
    print(f"\nCreating visualizations in: {FLAGS.output_dir}")
    
    # Create static analysis plots
    if FLAGS.save_static_plots:
        print("\nGenerating static analysis plots...")
        create_static_analysis_plots(
            data, FLAGS.output_dir, 
            plot_every_n=FLAGS.plot_every_n_steps,
            cvar_risk=FLAGS.cvar_risk
        )
    
    # Create animated visualization
    if FLAGS.create_animation and len(data['frames']) > 0:
        print("\nGenerating animated visualization...")
        create_animated_visualization(data, FLAGS.output_dir, fps=5)
    elif FLAGS.create_animation:
        print("Warning: No frames available for animation")
    
    print(f"\nVisualization complete! Check outputs in: {FLAGS.output_dir}")


if __name__ == "__main__":
    app.run(main)