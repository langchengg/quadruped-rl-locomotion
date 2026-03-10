"""
MuJoCo Gymnasium Environment for Unitree Go2 Quadruped Locomotion.

Migrated from the Genesis-based Argo-Robot/quadrupeds_locomotion project
to standard MuJoCo + Gymnasium interface. Key differences:
  - Uses mujoco v3+ Python bindings instead of Genesis
  - Standard Gymnasium API (reset/step/render)
  - Modular reward system (via rewards.py)
  - Domain randomization support (via domain_rand.py)
  - Position-controlled actuators with residual action design

Observation space (48-dim):
  [0:3]   Base angular velocity (scaled)
  [3:6]   Projected gravity vector
  [6:11]  Commands [vx, vy, wz, height, jump] (scaled)
  [11:23] Joint positions relative to default (12 joints)
  [23:35] Joint velocities (scaled) (12 joints)
  [35:47] Previous actions (12 joints)
  [47]    Jump phase indicator

Action space (12-dim):
  Residual joint positions added to default standing pose.
  target_pos = default_pos + action * action_scale
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any

from src.envs.rewards import compute_rewards, DEFAULT_REWARD_SCALES
from src.envs.domain_rand import DomainRandomizer, DomainRandomizationConfig


# ──────────────────────────────────────────────────
# Helper: quaternion operations (pure numpy)
# ──────────────────────────────────────────────────

def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by the inverse of quaternion q.
    
    MuJoCo quaternion format: [w, x, y, z]
    """
    w, x, y, z = q
    # Inverse of unit quaternion is conjugate
    q_inv = np.array([w, -x, -y, -z])
    
    # v as quaternion: [0, vx, vy, vz]
    v_q = np.array([0.0, v[0], v[1], v[2]])
    
    # q_inv * v_q * q
    result = quat_multiply(quat_multiply(q_inv, v_q), q)
    return result[1:]  # return xyz


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to Euler angles [roll, pitch, yaw]."""
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


# ──────────────────────────────────────────────────
# Main Environment
# ──────────────────────────────────────────────────

class QuadrupedEnv(gym.Env):
    """Gymnasium environment for quadruped locomotion in MuJoCo.
    
    This environment loads the Unitree Go2 robot model and trains
    it to walk using reinforcement learning. The design follows the
    Argo-Robot project's approach but uses standard MuJoCo bindings.
    
    Args:
        env_cfg: Environment configuration dict
        reward_cfg: Reward configuration dict
        command_cfg: Command sampling configuration
        domain_rand_cfg: Domain randomization config (None to disable)
        render_mode: "human" for viewer, "rgb_array" for video
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    # Default configurations (can be overridden via YAML)
    DEFAULT_ENV_CFG = {
        "num_actions": 12,
        "default_joint_angles": {
            # From Go2 MuJoCo menagerie home keyframe
            "FL_hip_joint": 0.0,    "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,    "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,  "FR_thigh_joint": 0.9,
            "RL_thigh_joint": 0.9,  "RR_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,  "FR_calf_joint": -1.8,
            "RL_calf_joint": -1.8,  "RR_calf_joint": -1.8,
        },
        "dof_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0,                              # PD position gain
        "kd": 0.5,                                # PD derivative gain
        "action_scale": 0.25,                     # Residual action scaling
        "clip_actions": 100.0,                    # Action clipping range
        "episode_length_s": 20.0,                 # Max episode length
        "resampling_time_s": 4.0,                 # Command resample interval
        "termination_if_roll_greater_than": 30.0, # degrees
        "termination_if_pitch_greater_than": 30.0,
        "base_init_pos": [0.0, 0.0, 0.27],       # From menagerie keyframe
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "simulate_action_latency": True,
    }
    
    DEFAULT_REWARD_CFG = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.27,
        "dt": 0.02,
        "reward_scales": DEFAULT_REWARD_SCALES.copy(),
    }
    
    DEFAULT_COMMAND_CFG = {
        "num_commands": 5,
        "lin_vel_x_range": [-0.5, 1.5],
        "lin_vel_y_range": [-0.3, 0.3],
        "ang_vel_range": [-0.5, 0.5],
        "height_range": [0.2, 0.35],
    }
    
    def __init__(self,
                 env_cfg: Optional[Dict] = None,
                 reward_cfg: Optional[Dict] = None,
                 command_cfg: Optional[Dict] = None,
                 domain_rand_cfg: Optional[DomainRandomizationConfig] = None,
                 render_mode: Optional[str] = None,
                 model_path: Optional[str] = None):
        super().__init__()
        
        # Merge configs with defaults
        self.env_cfg = {**self.DEFAULT_ENV_CFG, **(env_cfg or {})}
        self.reward_cfg = {**self.DEFAULT_REWARD_CFG, **(reward_cfg or {})}
        self.command_cfg = {**self.DEFAULT_COMMAND_CFG, **(command_cfg or {})}
        self.render_mode = render_mode
        
        # ─── Simulation parameters ───
        self.dt = 0.02  # 50Hz control
        self.sim_substeps = 4  # physics substeps per control step
        self.num_actions = self.env_cfg["num_actions"]
        self.max_episode_steps = int(self.env_cfg["episode_length_s"] / self.dt)
        self.action_scale = self.env_cfg["action_scale"]
        
        # ─── Load MuJoCo model ───
        if model_path is None:
            # Find the model relative to this file
            this_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(this_dir))
            model_path = os.path.join(project_root, "assets", "unitree_go2", "scene.xml")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.dt / self.sim_substeps
        self.data = mujoco.MjData(self.model)
        
        # ─── Joint mapping ───
        self.motor_dof_indices = []
        for name in self.env_cfg["dof_names"]:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            dof_idx = self.model.jnt_dofadr[joint_id]
            self.motor_dof_indices.append(dof_idx)
        
        self.motor_qpos_indices = []
        for name in self.env_cfg["dof_names"]:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_idx = self.model.jnt_qposadr[joint_id]
            self.motor_qpos_indices.append(qpos_idx)
        
        # Actuator indices
        self.actuator_indices = list(range(self.model.nu))
        
        # ─── Default joint positions ───
        self.default_dof_pos = np.array([
            self.env_cfg["default_joint_angles"][name]
            for name in self.env_cfg["dof_names"]
        ], dtype=np.float64)
        
        # ─── Observation and action spaces ───
        self.num_obs = 48
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_actions,), dtype=np.float32
        )
        
        # ─── Observation scales (from Argo-Robot project) ───
        self.obs_scales = {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        }
        self.commands_scale = np.array([
            self.obs_scales["lin_vel"],  # vx
            self.obs_scales["lin_vel"],  # vy
            self.obs_scales["ang_vel"],  # wz
            self.obs_scales["lin_vel"],  # height
            self.obs_scales["lin_vel"],  # jump
        ], dtype=np.float32)
        
        # ─── State buffers ───
        self.actions = np.zeros(self.num_actions, dtype=np.float64)
        self.last_actions = np.zeros(self.num_actions, dtype=np.float64)
        self.commands = np.zeros(5, dtype=np.float64)
        self.dof_pos = np.zeros(self.num_actions, dtype=np.float64)
        self.dof_vel = np.zeros(self.num_actions, dtype=np.float64)
        self.last_dof_vel = np.zeros(self.num_actions, dtype=np.float64)
        self.base_pos = np.zeros(3, dtype=np.float64)
        self.base_quat = np.array([1, 0, 0, 0], dtype=np.float64)
        self.base_lin_vel = np.zeros(3, dtype=np.float64)
        self.base_ang_vel = np.zeros(3, dtype=np.float64)
        self.projected_gravity = np.zeros(3, dtype=np.float64)
        self.base_euler = np.zeros(3, dtype=np.float64)
        
        self.episode_step = 0
        self.episode_rewards = {}
        
        # ─── Domain randomization ───
        self.domain_rand = None
        self.rand_params = {}
        if domain_rand_cfg is not None:
            self.domain_rand = DomainRandomizer(domain_rand_cfg)
            self.domain_rand.store_defaults(self.model)
        
        # ─── Renderer ───
        self.renderer = None
        if render_mode == "human":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        elif render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
    
    def _get_obs(self) -> np.ndarray:
        """Compute 48-dimensional observation vector."""
        obs = np.concatenate([
            self.base_ang_vel * self.obs_scales["ang_vel"],        # 3
            self.projected_gravity,                                  # 3
            self.commands * self.commands_scale,                     # 5
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            self.dof_vel * self.obs_scales["dof_vel"],              # 12
            self.actions,                                            # 12
            [0.0],                                                   # 1 (jump phase placeholder)
        ]).astype(np.float32)
        
        # Apply sensor noise (if domain randomization enabled)
        if self.domain_rand is not None:
            obs = self.domain_rand.apply_observation_noise(obs, self.rand_params)
        
        return obs
    
    def _update_state_from_sim(self):
        """Read state from MuJoCo simulation data."""
        # Base position and orientation
        self.base_pos = self.data.qpos[:3].copy()
        self.base_quat = self.data.qpos[3:7].copy()
        
        # Base velocities in body frame
        base_vel_world = self.data.qvel[:3].copy()
        base_angvel_world = self.data.qvel[3:6].copy()
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, base_vel_world)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, base_angvel_world)
        
        # Projected gravity (gravity in body frame)
        gravity_world = np.array([0.0, 0.0, -1.0])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, gravity_world)
        
        # Euler angles
        self.base_euler = quat_to_euler(self.base_quat)
        
        # Joint positions and velocities
        for i, qpos_idx in enumerate(self.motor_qpos_indices):
            self.dof_pos[i] = self.data.qpos[qpos_idx]
        for i, dof_idx in enumerate(self.motor_dof_indices):
            self.dof_vel[i] = self.data.qvel[dof_idx]
    
    def _sample_commands(self):
        """Sample random velocity commands for training."""
        cfg = self.command_cfg
        self.commands[0] = np.random.uniform(*cfg["lin_vel_x_range"])
        self.commands[1] = np.random.uniform(*cfg["lin_vel_y_range"])
        self.commands[2] = np.random.uniform(*cfg["ang_vel_range"])
        self.commands[3] = np.random.uniform(*cfg["height_range"])
        self.commands[4] = 0.0  # jump command (disabled for basic training)
    
    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should end.
        
        Returns:
            (terminated, truncated): terminated if unhealthy,
            truncated if max steps reached
        """
        roll_deg = np.degrees(abs(self.base_euler[0]))
        pitch_deg = np.degrees(abs(self.base_euler[1]))
        
        roll_limit = self.env_cfg["termination_if_roll_greater_than"]
        pitch_limit = self.env_cfg["termination_if_pitch_greater_than"]
        
        terminated = (roll_deg > roll_limit or pitch_deg > pitch_limit)
        truncated = (self.episode_step >= self.max_episode_steps)
        
        return terminated, truncated
    
    def _get_reward_state(self) -> Dict[str, Any]:
        """Package environment state for reward computation."""
        return {
            "base_lin_vel": self.base_lin_vel,
            "base_ang_vel": self.base_ang_vel,
            "base_pos": self.base_pos,
            "base_euler": self.base_euler,
            "projected_gravity": self.projected_gravity,
            "commands": self.commands,
            "dof_pos": self.dof_pos,
            "dof_vel": self.dof_vel,
            "last_dof_vel": self.last_dof_vel,
            "actions": self.actions,
            "last_actions": self.last_actions,
            "default_dof_pos": self.default_dof_pos,
            "torques": self.data.actuator_force[:self.num_actions].copy(),
        }
    
    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.
        
        Returns:
            (observation, info) tuple
        """
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial pose from keyframe or config
        init_pos = np.array(self.env_cfg["base_init_pos"])
        init_quat = np.array(self.env_cfg["base_init_quat"])
        
        # Set base position + orientation (first 7 qpos elements)
        self.data.qpos[:3] = init_pos
        self.data.qpos[3:7] = init_quat
        
        # Set default joint positions
        for i, qpos_idx in enumerate(self.motor_qpos_indices):
            self.data.qpos[qpos_idx] = self.default_dof_pos[i]
        
        # Add small random noise to initial pose
        noise_pos = np.random.uniform(-0.01, 0.01, len(self.motor_qpos_indices))
        for i, qpos_idx in enumerate(self.motor_qpos_indices):
            self.data.qpos[qpos_idx] += noise_pos[i]
        
        # Zero velocities
        self.data.qvel[:] = 0.0
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Reset buffers
        self.actions[:] = 0.0
        self.last_actions[:] = 0.0
        self.last_dof_vel[:] = 0.0
        self.episode_step = 0
        self.episode_rewards = {}
        
        # Sample initial commands
        self._sample_commands()
        
        # Apply domain randomization
        if self.domain_rand is not None:
            self.domain_rand.restore_defaults(self.model)
            self.rand_params = self.domain_rand.randomize(self.model, self.data)
        
        # Update state
        self._update_state_from_sim()
        
        obs = self._get_obs()
        info = {"commands": self.commands.copy()}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step.
        
        Args:
            action: 12-dim array of residual joint position commands
            
        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # Clip actions
        self.actions = np.clip(
            action.astype(np.float64),
            -self.env_cfg["clip_actions"],
            self.env_cfg["clip_actions"]
        )
        
        # Residual action: target = default + action * scale
        # Use last actions if simulating latency
        exec_actions = self.last_actions if self.env_cfg["simulate_action_latency"] else self.actions
        target_dof_pos = exec_actions * self.action_scale + self.default_dof_pos
        
        # Apply PD control via actuators
        kp = self.env_cfg["kp"]
        kd = self.env_cfg["kd"]
        
        # Compute torques: τ = kp * (target - pos) - kd * vel
        for i in range(self.num_actions):
            dof_idx = self.motor_dof_indices[i]
            qpos_idx = self.motor_qpos_indices[i]
            error = target_dof_pos[i] - self.data.qpos[qpos_idx]
            vel = self.data.qvel[dof_idx]
            torque = kp * error - kd * vel
            self.data.ctrl[i] = torque
        
        # Step simulation
        for _ in range(self.sim_substeps):
            mujoco.mj_step(self.model, self.data)
        
        # Apply external forces (domain randomization)
        if self.domain_rand is not None:
            self.domain_rand.apply_external_force(
                self.model, self.data, self.rand_params
            )
        
        # Update episode step
        self.episode_step += 1
        
        # Resample commands periodically
        resample_interval = int(self.env_cfg["resampling_time_s"] / self.dt)
        if self.episode_step % resample_interval == 0:
            self._sample_commands()
        
        # Update state variables
        self.last_dof_vel[:] = self.dof_vel[:]
        self._update_state_from_sim()
        
        # Compute rewards
        state = self._get_reward_state()
        reward_scales = self.reward_cfg.get("reward_scales", DEFAULT_REWARD_SCALES)
        rewards = compute_rewards(state, reward_scales, self.reward_cfg)
        
        total_reward = rewards.pop("total")
        
        # Accumulate episode reward stats
        for name, val in rewards.items():
            self.episode_rewards[name] = self.episode_rewards.get(name, 0.0) + val
        
        # Check termination
        terminated, truncated = self._check_termination()
        
        # Update action buffer
        self.last_actions[:] = self.actions[:]
        
        # Build info
        info = {
            "rewards": rewards,
            "commands": self.commands.copy(),
            "base_pos": self.base_pos.copy(),
            "base_euler_deg": np.degrees(self.base_euler),
        }
        
        if terminated or truncated:
            info["episode_rewards"] = self.episode_rewards.copy()
            info["episode_length"] = self.episode_step
        
        obs = self._get_obs()
        
        return obs, float(total_reward), terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.renderer is None:
            return None
        
        self.renderer.update_scene(self.data)
        img = self.renderer.render()
        
        return img
    
    def close(self):
        """Clean up renderer resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# ──────────────────────────────────────────────────
# Factory function for easy creation
# ──────────────────────────────────────────────────

def make_quadruped_env(
    render_mode: Optional[str] = None,
    domain_rand: bool = False,
    reward_scales: Optional[Dict[str, float]] = None,
    model_path: Optional[str] = None,
    **env_kwargs
) -> QuadrupedEnv:
    """Create a QuadrupedEnv with common presets.
    
    Args:
        render_mode: "human" or "rgb_array" or None
        domain_rand: Enable domain randomization
        reward_scales: Override default reward scales
        model_path: Override model path
        **env_kwargs: Override env_cfg params
    
    Returns:
        Configured QuadrupedEnv instance
    """
    env_cfg = {**QuadrupedEnv.DEFAULT_ENV_CFG, **env_kwargs}
    
    reward_cfg = QuadrupedEnv.DEFAULT_REWARD_CFG.copy()
    if reward_scales is not None:
        reward_cfg["reward_scales"] = reward_scales
    
    dr_cfg = DomainRandomizationConfig(enabled=True) if domain_rand else None
    
    return QuadrupedEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        domain_rand_cfg=dr_cfg,
        render_mode=render_mode,
        model_path=model_path,
    )
