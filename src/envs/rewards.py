"""
Modular reward functions for quadruped locomotion training.

Each reward function takes the environment state as input and returns
a scalar reward value. Rewards can be enabled/disabled and scaled
through the config YAML files for ablation experiments.

Design is inspired by Argo-Robot/quadrupeds_locomotion but adapted
for MuJoCo + Gymnasium interface with additional reward terms.
"""

import numpy as np
from typing import Dict, Callable, Any


# ──────────────────────────────────────────────────
# Core Tracking Rewards
# ──────────────────────────────────────────────────

def reward_tracking_lin_vel(state: Dict[str, Any], cfg: Dict) -> float:
    """Encourage tracking commanded linear velocity (xy plane).
    
    Uses exponential kernel: exp(-error² / σ) to give smooth gradient.
    σ (tracking_sigma) controls how "forgiving" the reward is.
    """
    cmd_vel = state["commands"][:2]  # [vx_ref, vy_ref]
    base_vel = state["base_lin_vel"][:2]
    error = np.sum((cmd_vel - base_vel) ** 2)
    sigma = cfg.get("tracking_sigma", 0.25)
    return float(np.exp(-error / sigma))


def reward_tracking_ang_vel(state: Dict[str, Any], cfg: Dict) -> float:
    """Encourage tracking commanded angular velocity (yaw).
    
    Similar exponential kernel as linear velocity tracking.
    """
    cmd_yaw = state["commands"][2]  # wz_ref
    base_yaw = state["base_ang_vel"][2]
    error = (cmd_yaw - base_yaw) ** 2
    sigma = cfg.get("tracking_sigma", 0.25)
    return float(np.exp(-error / sigma))


# ──────────────────────────────────────────────────
# Stability Penalties
# ──────────────────────────────────────────────────

def reward_lin_vel_z(state: Dict[str, Any], cfg: Dict) -> float:
    """Penalize vertical (z) body velocity — discourages bouncing."""
    vz = state["base_lin_vel"][2]
    return float(vz ** 2)


def reward_ang_vel_xy(state: Dict[str, Any], cfg: Dict) -> float:
    """Penalize roll/pitch rotational velocity — promotes smooth torso."""
    wx, wy = state["base_ang_vel"][0], state["base_ang_vel"][1]
    return float(wx ** 2 + wy ** 2)


def reward_orientation(state: Dict[str, Any], cfg: Dict) -> float:
    """Penalize deviation from upright orientation (roll & pitch).
    
    Uses projected gravity vector: if robot is upright, gravity
    projects purely onto -z. Any x/y component means tilting.
    """
    projected_gravity = state["projected_gravity"]
    return float(projected_gravity[0] ** 2 + projected_gravity[1] ** 2)


def reward_base_height(state: Dict[str, Any], cfg: Dict) -> float:
    """Penalize deviation from target base height.
    
    The target height can come from either a fixed config value
    or from the command (for variable height tracking).
    """
    z = state["base_pos"][2]
    target = state["commands"][3] if len(state["commands"]) > 3 else cfg.get("base_height_target", 0.3)
    return float((z - target) ** 2)


# ──────────────────────────────────────────────────
# Action Smoothness Penalties
# ──────────────────────────────────────────────────

def reward_action_rate(state: Dict[str, Any], cfg: Dict) -> float:
    """Penalize large changes between consecutive actions.
    
    Encourages smooth motor commands, reducing mechanical wear
    and producing more natural gaits.
    """
    actions = state["actions"]
    last_actions = state["last_actions"]
    return float(np.sum((actions - last_actions) ** 2))


def reward_torque(state: Dict[str, Any], cfg: Dict) -> float:
    """Penalize high torques — encourages energy-efficient locomotion."""
    torques = state.get("torques", np.zeros(12))
    return float(np.sum(torques ** 2))


def reward_joint_acceleration(state: Dict[str, Any], cfg: Dict) -> float:
    """Penalize large joint accelerations for smoother motion."""
    dof_vel = state["dof_vel"]
    last_dof_vel = state["last_dof_vel"]
    dt = cfg.get("dt", 0.02)
    acc = (dof_vel - last_dof_vel) / dt
    return float(np.sum(acc ** 2))


# ──────────────────────────────────────────────────
# Pose Regularization
# ──────────────────────────────────────────────────

def reward_similar_to_default(state: Dict[str, Any], cfg: Dict) -> float:
    """Penalize deviation from default standing pose.
    
    Keeps the robot near a known-good configuration,
    simplifying exploration during early training.
    """
    dof_pos = state["dof_pos"]
    default_pos = state["default_dof_pos"]
    return float(np.sum(np.abs(dof_pos - default_pos)))


def reward_joint_limits(state: Dict[str, Any], cfg: Dict) -> float:
    """Penalize joint positions near limits."""
    dof_pos = state["dof_pos"]
    joint_lower = state.get("joint_lower", np.full(12, -np.pi))
    joint_upper = state.get("joint_upper", np.full(12, np.pi))
    
    margin = cfg.get("joint_limit_margin", 0.1)
    below = np.maximum(joint_lower + margin - dof_pos, 0.0)
    above = np.maximum(dof_pos - joint_upper + margin, 0.0)
    return float(np.sum(below ** 2 + above ** 2))


# ──────────────────────────────────────────────────
# Contact & Gait Rewards
# ──────────────────────────────────────────────────

def reward_feet_contact_forces(state: Dict[str, Any], cfg: Dict) -> float:
    """Penalize excessive contact forces on feet."""
    contact_forces = state.get("feet_contact_forces", np.zeros(4))
    max_force = cfg.get("max_contact_force", 100.0)
    excess = np.maximum(contact_forces - max_force, 0.0)
    return float(np.sum(excess ** 2))


def reward_feet_air_time(state: Dict[str, Any], cfg: Dict) -> float:
    """Reward feet spending appropriate time in the air.
    
    Encourages proper swing phase duration for natural gaits.
    Only active when the robot is commanded to move.
    """
    air_time = state.get("feet_air_time", np.zeros(4))
    contact = state.get("feet_contact", np.ones(4, dtype=bool))
    cmd_vel_norm = np.linalg.norm(state["commands"][:2])
    
    if cmd_vel_norm < 0.1:
        return 0.0
    
    # Reward first contact after air phase
    target_air_time = cfg.get("target_air_time", 0.2)
    rew = np.sum((air_time - target_air_time) * contact.astype(float))
    return float(np.clip(rew, -1.0, 1.0))


# ──────────────────────────────────────────────────
# Forward Progress Reward (bonus)
# ──────────────────────────────────────────────────

def reward_forward_velocity(state: Dict[str, Any], cfg: Dict) -> float:
    """Direct reward for forward velocity — simple baseline.
    
    Use this instead of tracking_lin_vel for simpler experiments.
    """
    vx = state["base_lin_vel"][0]
    target = cfg.get("target_forward_vel", 1.0)
    return float(np.exp(-((vx - target) ** 2) / 0.25))


# ──────────────────────────────────────────────────
# Reward Registry
# ──────────────────────────────────────────────────

REWARD_REGISTRY: Dict[str, Callable] = {
    # Tracking (positive rewards)
    "tracking_lin_vel":     reward_tracking_lin_vel,
    "tracking_ang_vel":     reward_tracking_ang_vel,
    "forward_velocity":     reward_forward_velocity,
    
    # Stability penalties (use negative scales)
    "lin_vel_z":            reward_lin_vel_z,
    "ang_vel_xy":           reward_ang_vel_xy,
    "orientation":          reward_orientation,
    "base_height":          reward_base_height,
    
    # Smoothness penalties (use negative scales)
    "action_rate":          reward_action_rate,
    "torque":               reward_torque,
    "joint_acceleration":   reward_joint_acceleration,
    
    # Regularization (use negative scales)
    "similar_to_default":   reward_similar_to_default,
    "joint_limits":         reward_joint_limits,
    
    # Contact/gait
    "feet_contact_forces":  reward_feet_contact_forces,
    "feet_air_time":        reward_feet_air_time,
}

# Default reward scales (matching Argo-Robot project baseline)
DEFAULT_REWARD_SCALES = {
    "tracking_lin_vel":     1.0,
    "tracking_ang_vel":     0.2,
    "lin_vel_z":           -1.0,
    "base_height":        -50.0,
    "action_rate":         -0.005,
    "similar_to_default":  -0.1,
}


def compute_rewards(state: Dict[str, Any], 
                    reward_scales: Dict[str, float],
                    cfg: Dict) -> Dict[str, float]:
    """Compute all enabled rewards and return individual + total.
    
    Args:
        state: Dictionary containing environment state
        reward_scales: Dict of {reward_name: scale}. Only rewards
                       present here will be computed.
        cfg: Reward configuration parameters
    
    Returns:
        Dict with individual reward values and "total" key
    """
    rewards = {}
    total = 0.0
    dt = cfg.get("dt", 0.02)
    
    for name, scale in reward_scales.items():
        if name in REWARD_REGISTRY:
            raw = REWARD_REGISTRY[name](state, cfg)
            scaled = raw * scale * dt
            rewards[name] = scaled
            total += scaled
    
    rewards["total"] = total
    return rewards
