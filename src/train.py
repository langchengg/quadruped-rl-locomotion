"""
Training script for quadruped locomotion with PPO / SAC / TD3.

Usage:
    # PPO (default)
    python src/train.py --algo ppo --total_timesteps 2000000

    # SAC (off-policy alternative)
    python src/train.py --algo sac --total_timesteps 1000000

    # With domain randomization
    python src/train.py --algo ppo --domain_rand --total_timesteps 2000000

    # Resume training
    python src/train.py --algo ppo --resume logs/ppo_go2_20240101/best_model.zip

    # Custom reward scales (for ablation)
    python src/train.py --config configs/reward_ablation.yaml
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from src.envs.quadruped_env import QuadrupedEnv, make_quadruped_env
from src.envs.domain_rand import DomainRandomizationConfig
from src.utils import load_config, save_config


ALGO_MAP = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def make_env(rank: int, 
             env_cfg: dict, 
             reward_cfg: dict,
             command_cfg: dict,
             domain_rand: bool = False,
             model_path: str = None,
             seed: int = 0):
    """Factory function for creating environments in SubprocVecEnv."""
    def _init():
        dr_cfg = DomainRandomizationConfig(enabled=True) if domain_rand else None
        env = QuadrupedEnv(
            env_cfg=env_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            domain_rand_cfg=dr_cfg,
            render_mode=None,
            model_path=model_path,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args):
    """Main training function."""
    # ─── Configuration ───
    if args.config:
        config = load_config(args.config)
        env_cfg = config.get("env", QuadrupedEnv.DEFAULT_ENV_CFG)
        reward_cfg = config.get("reward", QuadrupedEnv.DEFAULT_REWARD_CFG)
        command_cfg = config.get("command", QuadrupedEnv.DEFAULT_COMMAND_CFG)
    else:
        env_cfg = QuadrupedEnv.DEFAULT_ENV_CFG.copy()
        reward_cfg = QuadrupedEnv.DEFAULT_REWARD_CFG.copy()
        command_cfg = QuadrupedEnv.DEFAULT_COMMAND_CFG.copy()
    
    # ─── Experiment naming ───
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"{args.algo}_go2_{timestamp}"
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"  Quadruped RL Locomotion Training")
    print(f"  Algorithm:     {args.algo.upper()}")
    print(f"  Timesteps:     {args.total_timesteps:,}")
    print(f"  Environments:  {args.num_envs}")
    print(f"  Domain Rand:   {args.domain_rand}")
    print(f"  Log dir:       {log_dir}")
    print(f"{'='*60}")
    
    # ─── Find model path ───
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "assets", "unitree_go2", "scene.xml")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please ensure the Go2 model is in assets/unitree_go2/")
        sys.exit(1)
    
    # ─── Create vectorized environments ───
    if args.num_envs > 1:
        env = SubprocVecEnv([
            make_env(i, env_cfg, reward_cfg, command_cfg, 
                    args.domain_rand, model_path, args.seed)
            for i in range(args.num_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(0, env_cfg, reward_cfg, command_cfg,
                    args.domain_rand, model_path, args.seed)
        ])
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # ─── Create evaluation environment ───
    eval_env = DummyVecEnv([
        make_env(99, env_cfg, reward_cfg, command_cfg,
                False, model_path, args.seed + 99)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # ─── Algorithm configuration ───
    algo_class = ALGO_MAP[args.algo]
    
    if args.algo == "ppo":
        algo_kwargs = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "max_grad_norm": 1.0,
            "policy_kwargs": {
                "net_arch": dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            },
            "verbose": 1,
            "tensorboard_log": log_dir,
            "device": args.device,
            "seed": args.seed,
        }
    elif args.algo == "sac":
        algo_kwargs = {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "ent_coef": "auto",
            "train_freq": 1,
            "gradient_steps": 1,
            "policy_kwargs": {
                "net_arch": [256, 256],
            },
            "verbose": 1,
            "tensorboard_log": log_dir,
            "device": args.device,
            "seed": args.seed,
        }
    elif args.algo == "td3":
        algo_kwargs = {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            "policy_kwargs": {
                "net_arch": [256, 256],
            },
            "verbose": 1,
            "tensorboard_log": log_dir,
            "device": args.device,
            "seed": args.seed,
        }
    
    # ─── Create or load model ───
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = algo_class.load(args.resume, env=env)
    else:
        model = algo_class("MlpPolicy", env, **algo_kwargs)
    
    # ─── Callbacks ───
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=max(10000 // args.num_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // args.num_envs, 1),
        save_path=log_dir,
        name_prefix="checkpoint",
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # ─── Save config ───
    full_config = {
        "algorithm": args.algo,
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "domain_rand": args.domain_rand,
        "seed": args.seed,
        "env": env_cfg,
        "reward": reward_cfg,
        "command": command_cfg,
        "algo_kwargs": {k: str(v) for k, v in algo_kwargs.items()},
    }
    save_config(full_config, os.path.join(log_dir, "config.yaml"))
    
    # ─── Train! ───
    print(f"\nStarting training for {args.total_timesteps:,} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/3600:.1f} hours")
    
    # ─── Save final model ───
    final_path = os.path.join(log_dir, "final_model")
    model.save(final_path)
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    print(f"Final model saved to {final_path}")
    print(f"VecNormalize stats saved to {log_dir}/vec_normalize.pkl")


def main():
    parser = argparse.ArgumentParser(
        description="Train quadruped locomotion with RL"
    )
    parser.add_argument(
        "--algo", type=str, default="ppo",
        choices=["ppo", "sac", "td3"],
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=2_000_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--num_envs", type=int, default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--domain_rand", action="store_true",
        help="Enable domain randomization"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--exp_name", type=str, default=None,
        help="Experiment name (auto-generated if not set)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to model checkpoint to resume from"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use: 'auto', 'cpu', or 'cuda'"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
