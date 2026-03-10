"""
Evaluation script for trained quadruped locomotion policies.

Usage:
    # Evaluate and save video
    python src/evaluate.py --model logs/ppo_go2/best_model.zip --save_video

    # Random policy test (no trained model needed)
    python src/evaluate.py --policy random --num_steps 200

    # With specific commands
    python src/evaluate.py --model logs/ppo_go2/best_model.zip --vx 1.0 --vy 0.0 --wz 0.0
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.quadruped_env import QuadrupedEnv, make_quadruped_env
from src.utils import save_video, save_gif


ALGO_MAP = {"ppo": PPO, "sac": SAC, "td3": TD3}


def evaluate(args):
    """Run policy evaluation with optional video saving."""
    # ─── Create environment ───
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "assets", "unitree_go2", "scene.xml")
    
    render_mode = "rgb_array" if args.save_video or args.save_gif else args.render_mode
    
    env = make_quadruped_env(
        render_mode=render_mode,
        domain_rand=False,
        model_path=model_path,
    )
    
    # ─── Load policy ───
    policy = None
    if args.model and args.policy != "random":
        # Detect algorithm from filename
        algo_name = args.algo
        algo_class = ALGO_MAP[algo_name]
        policy = algo_class.load(args.model)
        print(f"Loaded {algo_name.upper()} policy from {args.model}")
    
    # ─── Evaluation loop ───
    frames = []
    total_reward = 0.0
    episode_count = 0
    step_count = 0
    
    obs, info = env.reset()
    
    # Set manual commands if specified
    if args.vx is not None:
        env.commands[0] = args.vx
    if args.vy is not None:
        env.commands[1] = args.vy
    if args.wz is not None:
        env.commands[2] = args.wz
    
    print(f"\nEvaluating for {args.num_steps} steps...")
    print(f"Commands: vx={env.commands[0]:.2f}, vy={env.commands[1]:.2f}, "
          f"wz={env.commands[2]:.2f}")
    
    for step in range(args.num_steps):
        # Get action
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Capture frame
        if render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        # Print progress
        if step % 100 == 0:
            pos = info.get("base_pos", env.base_pos)
            euler = info.get("base_euler_deg", np.degrees(env.base_euler))
            print(f"  Step {step:4d} | "
                  f"Reward: {reward:7.3f} | "
                  f"Pos: [{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}] | "
                  f"R/P/Y: [{euler[0]:5.1f}°, {euler[1]:5.1f}°, {euler[2]:5.1f}°]")
        
        if terminated or truncated:
            episode_count += 1
            if episode_count >= args.num_episodes:
                break
            obs, info = env.reset()
            if args.vx is not None:
                env.commands[0] = args.vx
            if args.vy is not None:
                env.commands[1] = args.vy
            if args.wz is not None:
                env.commands[2] = args.wz
    
    # ─── Summary ───
    print(f"\n{'='*50}")
    print(f"  Evaluation Summary")
    print(f"  Steps:     {step_count}")
    print(f"  Episodes:  {max(episode_count, 1)}")
    print(f"  Total Rew: {total_reward:.2f}")
    print(f"  Avg Rew:   {total_reward / max(step_count, 1):.4f}")
    print(f"{'='*50}")
    
    # ─── Save video/GIF ───
    if frames:
        os.makedirs("results/videos", exist_ok=True)
        if args.save_video:
            video_path = args.output or "results/videos/evaluation.mp4"
            save_video(frames, video_path, fps=50)
        if args.save_gif:
            gif_path = args.output or "results/videos/evaluation.gif"
            # Subsample for smaller GIF
            skip = max(1, len(frames) // 200)
            save_gif(frames[::skip], gif_path, fps=20)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained quadruped locomotion policy"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model file (.zip)"
    )
    parser.add_argument(
        "--algo", type=str, default="ppo",
        choices=["ppo", "sac", "td3"],
        help="Algorithm used for training"
    )
    parser.add_argument(
        "--policy", type=str, default="trained",
        choices=["trained", "random"],
        help="Policy type: 'trained' (requires --model) or 'random'"
    )
    parser.add_argument(
        "--num_steps", type=int, default=1000,
        help="Number of evaluation steps"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=5,
        help="Max number of episodes"
    )
    parser.add_argument(
        "--render_mode", type=str, default=None,
        choices=["human", "rgb_array", None],
        help="Render mode"
    )
    parser.add_argument(
        "--save_video", action="store_true",
        help="Save evaluation as MP4 video"
    )
    parser.add_argument(
        "--save_gif", action="store_true",
        help="Save evaluation as GIF"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for video/gif"
    )
    # Command overrides
    parser.add_argument("--vx", type=float, default=None, help="Forward velocity command")
    parser.add_argument("--vy", type=float, default=None, help="Lateral velocity command")
    parser.add_argument("--wz", type=float, default=None, help="Yaw velocity command")
    
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
