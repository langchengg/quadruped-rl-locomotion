"""
Terrain generation for MuJoCo heightfield-based environments.

Creates various terrain types (flat, slopes, stairs, rough) and
supports curriculum learning by gradually increasing difficulty.
Terrain is injected by modifying the MuJoCo XML before loading.
"""

import numpy as np
from typing import Optional, Tuple


class TerrainGenerator:
    """Generates heightfield terrain data for MuJoCo environments.
    
    MuJoCo heightfields are 2D grids of elevation values.
    This class creates various terrain patterns and supports
    difficulty-based curriculum learning.
    
    Attributes:
        nrow: Number of rows in the heightfield
        ncol: Number of columns in the heightfield
        size: Physical size [x, y, z_max, z_offset] of the heightfield
        resolution: Grid spacing in meters
    """
    
    def __init__(self, 
                 nrow: int = 200, 
                 ncol: int = 200, 
                 size: Tuple[float, float, float, float] = (10.0, 10.0, 0.5, 0.0),
                 resolution: float = 0.05):
        self.nrow = nrow
        self.ncol = ncol
        self.size = size
        self.resolution = resolution
        self.heightfield = np.zeros((nrow, ncol), dtype=np.float32)
    
    def flat(self) -> np.ndarray:
        """Generate a flat terrain (baseline)."""
        self.heightfield = np.zeros((self.nrow, self.ncol), dtype=np.float32)
        return self.heightfield
    
    def slope(self, 
              angle_deg: float = 5.0, 
              direction: str = "x") -> np.ndarray:
        """Generate a sloped terrain.
        
        Args:
            angle_deg: Slope angle in degrees
            direction: Slope direction ('x' or 'y')
        """
        angle_rad = np.radians(angle_deg)
        if direction == "x":
            x = np.linspace(0, self.size[0], self.ncol)
            heights = np.tan(angle_rad) * x
            self.heightfield = np.tile(heights, (self.nrow, 1)).astype(np.float32)
        else:
            y = np.linspace(0, self.size[1], self.nrow)
            heights = np.tan(angle_rad) * y
            self.heightfield = np.tile(heights.reshape(-1, 1), (1, self.ncol)).astype(np.float32)
        
        # Normalize to [0, 1] for MuJoCo heightfield
        if self.heightfield.max() > 0:
            self.heightfield /= self.heightfield.max()
        return self.heightfield
    
    def stairs(self, 
               step_height: float = 0.05, 
               step_width: float = 0.3,
               direction: str = "x") -> np.ndarray:
        """Generate stair-like terrain.
        
        Args:
            step_height: Height of each step in meters
            step_width: Width of each step in meters
        """
        step_width_cells = max(1, int(step_width / self.resolution))
        n_steps = self.ncol // step_width_cells
        
        for i in range(n_steps):
            start = i * step_width_cells
            end = min(start + step_width_cells, self.ncol)
            height = i * step_height
            self.heightfield[:, start:end] = height
        
        # Normalize
        max_h = self.heightfield.max()
        if max_h > 0:
            self.heightfield /= max_h
        return self.heightfield
    
    def random_rough(self, 
                     amplitude: float = 0.02,
                     frequency: float = 1.0,
                     seed: Optional[int] = None) -> np.ndarray:
        """Generate random rough terrain using Perlin-like noise.
        
        Args:
            amplitude: Maximum height variation in meters
            frequency: Controls roughness (higher = rougher)
            seed: Random seed for reproducibility
        """
        rng = np.random.RandomState(seed)
        
        # Multi-scale noise for natural-looking terrain
        self.heightfield = np.zeros((self.nrow, self.ncol), dtype=np.float32)
        
        for octave in range(3):
            scale = frequency * (2 ** octave)
            weight = amplitude / (2 ** octave)
            noise = rng.randn(
                max(1, int(self.nrow / scale)), 
                max(1, int(self.ncol / scale))
            ) * weight
            
            # Upsample to full resolution using bilinear interpolation
            from scipy.ndimage import zoom
            zoom_factor_r = self.nrow / noise.shape[0]
            zoom_factor_c = self.ncol / noise.shape[1]
            upsampled = zoom(noise, (zoom_factor_r, zoom_factor_c), order=1)
            
            # Handle shape mismatch from rounding
            self.heightfield[:upsampled.shape[0], :upsampled.shape[1]] += upsampled[
                :self.nrow, :self.ncol
            ].astype(np.float32)
        
        # Normalize to [0, 1]
        h_min, h_max = self.heightfield.min(), self.heightfield.max()
        if h_max - h_min > 0:
            self.heightfield = (self.heightfield - h_min) / (h_max - h_min)
        
        return self.heightfield
    
    def mixed(self, 
              difficulty: float = 0.5,
              seed: Optional[int] = None) -> np.ndarray:
        """Generate mixed terrain with patches of different types.
        
        Splits the terrain into horizontal strips with different patterns.
        
        Args:
            difficulty: 0.0 = mostly flat, 1.0 = challenging terrain
            seed: Random seed
        """
        rng = np.random.RandomState(seed)
        n_strips = 4
        strip_width = self.ncol // n_strips
        
        self.heightfield = np.zeros((self.nrow, self.ncol), dtype=np.float32)
        
        # Strip 1: Flat
        # (already zero)
        
        # Strip 2: Gentle slope
        if difficulty > 0.2:
            angle = 3.0 + difficulty * 12.0
            x = np.linspace(0, 1, strip_width)
            slope_heights = np.tan(np.radians(angle)) * x * self.resolution * strip_width
            self.heightfield[:, strip_width:2*strip_width] = np.tile(
                slope_heights, (self.nrow, 1)
            )
        
        # Strip 3: Stairs
        if difficulty > 0.4:
            step_h = 0.02 + difficulty * 0.08
            step_w_cells = max(1, int(0.3 / self.resolution))
            for i in range(strip_width // step_w_cells):
                start = 2 * strip_width + i * step_w_cells
                end = min(start + step_w_cells, 3 * strip_width)
                self.heightfield[:, start:end] = i * step_h
        
        # Strip 4: Rough terrain
        if difficulty > 0.3:
            amp = 0.01 + difficulty * 0.04
            noise = rng.randn(self.nrow, strip_width) * amp
            self.heightfield[:, 3*strip_width:] = noise[:, :self.ncol - 3*strip_width]
        
        # Normalize
        h_min, h_max = self.heightfield.min(), self.heightfield.max()
        if h_max - h_min > 0:
            self.heightfield = (self.heightfield - h_min) / (h_max - h_min)
        
        return self.heightfield
    
    def curriculum(self, difficulty_level: float, seed: Optional[int] = None) -> np.ndarray:
        """Generate terrain based on training curriculum.
        
        Difficulty progression:
          0.0 - 0.3: Flat terrain (learn basic balance)
          0.3 - 0.6: Gentle slopes (learn adaptation)
          0.6 - 0.8: Stairs and moderate slopes (leg coordination)
          0.8 - 1.0: Random rough terrain (full generalization)
        
        Args:
            difficulty_level: Training difficulty in [0, 1]
            seed: Random seed for reproducibility
        """
        if difficulty_level < 0.3:
            return self.flat()
        elif difficulty_level < 0.6:
            angle = 3.0 + (difficulty_level - 0.3) / 0.3 * 12.0
            return self.slope(angle_deg=angle)
        elif difficulty_level < 0.8:
            step_height = 0.03 + (difficulty_level - 0.6) / 0.2 * 0.07
            return self.stairs(step_height=step_height)
        else:
            amplitude = 0.02 + (difficulty_level - 0.8) / 0.2 * 0.05
            return self.random_rough(amplitude=amplitude, seed=seed)
    
    def get_heightfield_xml_snippet(self) -> str:
        """Generate MuJoCo XML snippet for the heightfield asset.
        
        Returns an XML string that can be injected into the model
        to add a heightfield terrain.
        """
        return f'''
    <asset>
        <hfield name="terrain" nrow="{self.nrow}" ncol="{self.ncol}" 
                size="{self.size[0]} {self.size[1]} {self.size[2]} {self.size[3]}"/>
    </asset>
    <worldbody>
        <geom name="floor_hfield" type="hfield" hfield="terrain" 
              pos="0 0 0" rgba="0.3 0.5 0.3 1"/>
    </worldbody>
    '''
