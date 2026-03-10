"""
Utility functions for training, evaluation, and visualization.
"""

import os
import yaml
import numpy as np
from typing import Dict, Any, Optional
import imageio


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_video(frames: list, path: str, fps: int = 30):
    """Save list of RGB frames as MP4 video.
    
    Args:
        frames: List of numpy arrays (H, W, 3)
        path: Output path (should end in .mp4)
        fps: Frames per second
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Video saved to {path} ({len(frames)} frames, {fps} fps)")


def save_gif(frames: list, path: str, fps: int = 20):
    """Save list of RGB frames as GIF.
    
    Args:
        frames: List of numpy arrays (H, W, 3)
        path: Output path (should end in .gif)
        fps: Frames per second
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    duration = 1000 / fps  # ms per frame
    imageio.mimsave(path, frames, duration=duration, loop=0)
    print(f"GIF saved to {path} ({len(frames)} frames, {fps} fps)")


def plot_training_curves(log_path: str, save_path: Optional[str] = None):
    """Plot training reward curves from SB3 logs.
    
    Args:
        log_path: Path to tensorboard log directory
        save_path: Where to save the plot (None = show)
    """
    try:
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        ea = EventAccumulator(log_path)
        ea.Reload()
        
        # Get episode reward data
        reward_events = ea.Scalars('rollout/ep_rew_mean')
        steps = [e.step for e in reward_events]
        rewards = [e.value for e in reward_events]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(steps, rewards, linewidth=2, color='#2196F3')
        ax.fill_between(steps, rewards, alpha=0.1, color='#2196F3')
        ax.set_xlabel('Timesteps', fontsize=12)
        ax.set_ylabel('Mean Episode Reward', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    except Exception as e:
        print(f"Could not plot training curves: {e}")
        print("Make sure tensorboard and matplotlib are installed.")


def plot_reward_ablation(results: Dict[str, list], save_path: Optional[str] = None):
    """Plot comparison of different reward ablation experiments.
    
    Args:
        results: Dict of {experiment_name: [rewards_list]}
        save_path: Where to save the plot
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for (name, rewards), color in zip(results.items(), colors):
        steps = np.arange(len(rewards))
        # Smooth with moving average
        window = min(50, len(rewards) // 10)
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(steps[:len(smoothed)], smoothed, 
                   label=name, color=color, linewidth=2)
        else:
            ax.plot(steps, rewards, label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Training Steps (×1000)', fontsize=12)
    ax.set_ylabel('Mean Episode Reward', fontsize=12)
    ax.set_title('Reward Ablation Study', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Ablation plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


class RewardLogger:
    """Tracks individual reward components across training."""
    
    def __init__(self):
        self.history: Dict[str, list] = {}
    
    def log(self, rewards: Dict[str, float]):
        """Log reward values from one step."""
        for name, val in rewards.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(val)
    
    def get_summary(self, last_n: int = 100) -> Dict[str, float]:
        """Get mean of each reward component over last N steps."""
        summary = {}
        for name, vals in self.history.items():
            recent = vals[-last_n:] if len(vals) >= last_n else vals
            summary[name] = np.mean(recent)
        return summary
    
    def save(self, path: str):
        """Save reward history to numpy file."""
        np.savez(path, **{k: np.array(v) for k, v in self.history.items()})
        print(f"Reward history saved to {path}")
