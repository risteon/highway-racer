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
    print(f"  Accelerate: {data['q_probs_accelerate'].shape}")
    print(f"  Brake: {data['q_probs_brake'].shape}")
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
    stats_accel = compute_q_statistics(data['q_probs_accelerate'], data['q_atoms_accelerate'], cvar_risk)
    stats_brake = compute_q_statistics(data['q_probs_brake'], data['q_atoms_brake'], cvar_risk)
    stats_policy = compute_q_statistics(data['q_probs_policy'], data['q_atoms_policy'], cvar_risk)
    
    # 1. Q-Value Evolution Over Time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Expected Q-values
    axes[0, 0].plot(stats_accel['ensemble_mean'], label='Accelerate', linewidth=2)
    axes[0, 0].plot(stats_brake['ensemble_mean'], label='Brake', linewidth=2)
    axes[0, 0].plot(stats_policy['ensemble_mean'], label='Policy Action', linewidth=2)
    axes[0, 0].fill_between(range(len(stats_accel['ensemble_mean'])), 
                           stats_accel['ensemble_mean'] - stats_accel['ensemble_std'],
                           stats_accel['ensemble_mean'] + stats_accel['ensemble_std'], alpha=0.3)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Expected Q-Value')
    axes[0, 0].set_title('Q-Value Evolution (Expected Value)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # CVaR Q-values (Risk-Sensitive)
    axes[0, 1].plot(np.mean(stats_accel['cvar'], axis=-1), label=f'Accelerate (CVaR {cvar_risk})', linewidth=2)
    axes[0, 1].plot(np.mean(stats_brake['cvar'], axis=-1), label=f'Brake (CVaR {cvar_risk})', linewidth=2)
    axes[0, 1].plot(np.mean(stats_policy['cvar'], axis=-1), label=f'Policy Action (CVaR {cvar_risk})', linewidth=2)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('CVaR Q-Value')
    axes[0, 1].set_title(f'Risk-Sensitive Q-Values (CVaR α={cvar_risk})')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Q-Value Uncertainty (Aleatoric)
    axes[1, 0].plot(stats_accel['aleatoric_mean'], label='Accelerate', linewidth=2)
    axes[1, 0].plot(stats_brake['aleatoric_mean'], label='Brake', linewidth=2)
    axes[1, 0].plot(stats_policy['aleatoric_mean'], label='Policy Action', linewidth=2)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Q-Value Std Dev')
    axes[1, 0].set_title('Aleatoric Uncertainty (Distribution Width)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Epistemic Uncertainty (Ensemble Disagreement)
    axes[1, 1].plot(stats_accel['ensemble_std'], label='Accelerate', linewidth=2)
    axes[1, 1].plot(stats_brake['ensemble_std'], label='Brake', linewidth=2)
    axes[1, 1].plot(stats_policy['ensemble_std'], label='Policy Action', linewidth=2)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Ensemble Std Dev')
    axes[1, 1].set_title('Epistemic Uncertainty (Ensemble Disagreement)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'q_value_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Q-Distribution Heatmaps for Selected Steps
    n_plots = min(6, len(steps_to_plot))
    fig, axes = plt.subplots(3, n_plots, figsize=(4*n_plots, 12))
    if n_plots == 1:
        axes = axes.reshape(-1, 1)
    
    for i, step in enumerate(list(steps_to_plot)[:n_plots]):
        # Get Q-distributions at this step (average over ensemble and squeeze batch dim)
        q_probs_accel = np.mean(data['q_probs_accelerate'][step].squeeze(), axis=0)
        q_atoms_accel = np.mean(data['q_atoms_accelerate'][step].squeeze(), axis=0)
        q_probs_brake = np.mean(data['q_probs_brake'][step].squeeze(), axis=0)
        q_atoms_brake = np.mean(data['q_atoms_brake'][step].squeeze(), axis=0)
        q_probs_policy = np.mean(data['q_probs_policy'][step].squeeze(), axis=0)
        q_atoms_policy = np.mean(data['q_atoms_policy'][step].squeeze(), axis=0)
        
        # Calculate bar width safely
        if len(q_atoms_accel) > 1:
            width_accel = (q_atoms_accel[1] - q_atoms_accel[0]) * 0.8
        else:
            width_accel = 1.0
        
        # Plot distributions
        axes[0, i].bar(q_atoms_accel, q_probs_accel, width=width_accel, alpha=0.7)
        axes[0, i].set_title(f'Accelerate (Step {step})')
        axes[0, i].set_xlabel('Q-Value')
        axes[0, i].set_ylabel('Probability')
        
        # Calculate bar widths safely for other plots
        if len(q_atoms_brake) > 1:
            width_brake = (q_atoms_brake[1] - q_atoms_brake[0]) * 0.8
        else:
            width_brake = 1.0
        if len(q_atoms_policy) > 1:
            width_policy = (q_atoms_policy[1] - q_atoms_policy[0]) * 0.8
        else:
            width_policy = 1.0
        
        axes[1, i].bar(q_atoms_brake, q_probs_brake, width=width_brake, alpha=0.7, color='orange')
        axes[1, i].set_title(f'Brake (Step {step})')
        axes[1, i].set_xlabel('Q-Value')
        axes[1, i].set_ylabel('Probability')
        
        axes[2, i].bar(q_atoms_policy, q_probs_policy, width=width_policy, alpha=0.7, color='green')
        axes[2, i].set_title(f'Policy Action (Step {step})')
        axes[2, i].set_xlabel('Q-Value')
        axes[2, i].set_ylabel('Probability')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'q_distributions_selected_steps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Action Preference Analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Q-value difference (Accelerate - Brake)
    q_diff_mean = stats_accel['ensemble_mean'] - stats_brake['ensemble_mean']
    q_diff_cvar = np.mean(stats_accel['cvar'], axis=-1) - np.mean(stats_brake['cvar'], axis=-1)
    
    ax1.plot(q_diff_mean, label='Expected Q-Value Difference', linewidth=2)
    ax1.plot(q_diff_cvar, label=f'CVaR Difference (α={cvar_risk})', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.fill_between(range(len(q_diff_mean)), 0, q_diff_mean, 
                     where=(q_diff_mean > 0), alpha=0.3, color='blue', label='Prefer Accelerate')
    ax1.fill_between(range(len(q_diff_mean)), 0, q_diff_mean, 
                     where=(q_diff_mean < 0), alpha=0.3, color='red', label='Prefer Brake')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Q-Value Difference')
    ax1.set_title('Action Preference: Accelerate vs Brake')
    ax1.legend()
    ax1.grid(True)
    
    # Actual actions taken
    actions = data['actions_taken']
    acceleration_actions = actions[:, 1]  # Second dimension is acceleration
    ax2.plot(acceleration_actions, label='Acceleration Actions', linewidth=2, color='purple')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Acceleration Action')
    ax2.set_title('Actual Actions Taken by Policy')
    ax2.legend()
    ax2.grid(True)
    
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