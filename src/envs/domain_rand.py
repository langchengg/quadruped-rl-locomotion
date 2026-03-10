"""
Domain Randomization module for sim-to-real transfer.

Randomizes physical parameters during training to produce policies
that generalize better to the real world. Parameters include:
  - Floor friction
  - Robot mass / inertia
  - Action latency
  - Sensor noise (IMU, joint encoders)
  - External disturbance forces

Inspired by Argo-Robot/quadrupeds_locomotion domain randomization
and the "domain randomization" strategy from Tan et al. (2018).
"""

import numpy as np
import mujoco
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization parameters.
    
    Each parameter has a (min, max) range. At each episode reset,
    values are sampled uniformly from these ranges.
    """
    enabled: bool = True
    
    # Floor friction coefficient [min, max]
    friction_range: Tuple[float, float] = (0.3, 1.2)
    
    # Robot base mass multiplier [min, max] (1.0 = default)
    mass_multiplier_range: Tuple[float, float] = (0.8, 1.2)
    
    # Action latency in timesteps [min, max]
    latency_range: Tuple[int, int] = (0, 2)
    
    # IMU (angular velocity) noise std [min, max]
    imu_noise_std_range: Tuple[float, float] = (0.0, 0.05)
    
    # Joint position encoder noise std [min, max]
    joint_pos_noise_std_range: Tuple[float, float] = (0.0, 0.02)
    
    # Joint velocity encoder noise std [min, max]
    joint_vel_noise_std_range: Tuple[float, float] = (0.0, 0.1)
    
    # External force perturbation magnitude [min, max] (N)
    ext_force_range: Tuple[float, float] = (0.0, 5.0)
    
    # Probability of applying external force at each step
    ext_force_prob: float = 0.005
    
    # Joint damping multiplier range
    damping_multiplier_range: Tuple[float, float] = (0.8, 1.2)
    
    # Motor strength (Kp) multiplier range
    kp_multiplier_range: Tuple[float, float] = (0.8, 1.2)


class DomainRandomizer:
    """Applies domain randomization to MuJoCo environments.
    
    Usage:
        rand = DomainRandomizer(config)
        # At each episode reset:
        params = rand.randomize(model, data)
        # At each step (for noise and forces):
        noisy_obs = rand.apply_observation_noise(obs, params)
        rand.apply_external_force(model, data, params)
    """
    
    def __init__(self, config: DomainRandomizationConfig, 
                 rng: Optional[np.random.RandomState] = None):
        self.config = config
        self.rng = rng or np.random.RandomState()
        
        # Store original model parameters for resetting
        self._original_params: Dict = {}
    
    def store_defaults(self, model: mujoco.MjModel):
        """Store the original model parameters before any randomization.
        
        Should be called once during environment initialization.
        """
        self._original_params = {
            "body_mass": model.body_mass.copy(),
            "body_inertia": model.body_inertia.copy(),
            "dof_damping": model.dof_damping.copy(),
            "geom_friction": model.geom_friction.copy(),
        }
    
    def randomize(self, model: mujoco.MjModel, 
                  data: mujoco.MjData) -> Dict:
        """Randomize model parameters at episode reset.
        
        Modifies the MuJoCo model in-place and returns the
        sampled parameter values for logging/analysis.
        
        Args:
            model: MuJoCo model to modify
            data: MuJoCo data (for force application)
        
        Returns:
            Dict of sampled parameter values
        """
        if not self.config.enabled:
            return {}
        
        cfg = self.config
        params = {}
        
        # 1. Floor Friction
        friction = self.rng.uniform(*cfg.friction_range)
        params["friction"] = friction
        # Apply to all ground-like geoms (geom 0 is typically floor)
        for i in range(model.ngeom):
            if self._original_params.get("geom_friction") is not None:
                model.geom_friction[i, 0] = (
                    self._original_params["geom_friction"][i, 0] * 
                    friction / 0.6  # normalize by default friction
                )
        
        # 2. Mass randomization
        mass_mult = self.rng.uniform(*cfg.mass_multiplier_range)
        params["mass_multiplier"] = mass_mult
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_body_id >= 0:
            model.body_mass[base_body_id] = (
                self._original_params["body_mass"][base_body_id] * mass_mult
            )
        
        # 3. Joint damping
        damping_mult = self.rng.uniform(*cfg.damping_multiplier_range)
        params["damping_multiplier"] = damping_mult
        model.dof_damping[:] = self._original_params["dof_damping"] * damping_mult
        
        # 4. Action latency (stored as parameter, applied in env)
        latency = self.rng.randint(cfg.latency_range[0], cfg.latency_range[1] + 1)
        params["latency"] = latency
        
        # 5. Sensor noise levels
        params["imu_noise_std"] = self.rng.uniform(*cfg.imu_noise_std_range)
        params["joint_pos_noise_std"] = self.rng.uniform(*cfg.joint_pos_noise_std_range)
        params["joint_vel_noise_std"] = self.rng.uniform(*cfg.joint_vel_noise_std_range)
        
        # 6. Kp multiplier
        kp_mult = self.rng.uniform(*cfg.kp_multiplier_range)
        params["kp_multiplier"] = kp_mult
        
        return params
    
    def restore_defaults(self, model: mujoco.MjModel):
        """Restore original model parameters."""
        if self._original_params:
            model.body_mass[:] = self._original_params["body_mass"]
            model.body_inertia[:] = self._original_params["body_inertia"]
            model.dof_damping[:] = self._original_params["dof_damping"]
            model.geom_friction[:] = self._original_params["geom_friction"]
    
    def apply_observation_noise(self, obs: np.ndarray, 
                                 params: Dict) -> np.ndarray:
        """Add sensor noise to observations.
        
        Noise structure (matching 48-dim observation space):
          [0:3]   base angular velocity  → IMU noise
          [3:6]   projected gravity      → IMU noise
          [6:11]  commands               → no noise (user input)
          [11:23] joint positions         → encoder noise
          [23:35] joint velocities        → encoder noise
          [35:47] actions                 → no noise
          [47]    jump phase              → no noise
        """
        if not self.config.enabled:
            return obs
        
        noisy_obs = obs.copy()
        
        # IMU noise on angular velocity and gravity
        imu_std = params.get("imu_noise_std", 0.0)
        if imu_std > 0:
            noisy_obs[:3] += self.rng.normal(0, imu_std, 3)
            noisy_obs[3:6] += self.rng.normal(0, imu_std, 3)
        
        # Joint position encoder noise
        pos_std = params.get("joint_pos_noise_std", 0.0)
        if pos_std > 0:
            noisy_obs[11:23] += self.rng.normal(0, pos_std, 12)
        
        # Joint velocity encoder noise
        vel_std = params.get("joint_vel_noise_std", 0.0)
        if vel_std > 0:
            noisy_obs[23:35] += self.rng.normal(0, vel_std, 12)
        
        return noisy_obs
    
    def apply_external_force(self, model: mujoco.MjModel, 
                              data: mujoco.MjData,
                              params: Dict):
        """Apply random external force perturbation to the robot base.
        
        Forces are applied with low probability to simulate
        unexpected pushes/pulls in the real world.
        """
        if not self.config.enabled:
            return
        
        if self.rng.random() < self.config.ext_force_prob:
            force_mag = self.rng.uniform(*self.config.ext_force_range)
            # Random direction in XY plane
            angle = self.rng.uniform(0, 2 * np.pi)
            force = np.array([
                force_mag * np.cos(angle),
                force_mag * np.sin(angle),
                0.0  # No vertical force
            ])
            
            base_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "base"
            )
            if base_body_id >= 0:
                data.xfrc_applied[base_body_id, :3] = force
        else:
            # Clear external forces
            data.xfrc_applied[:, :] = 0.0
