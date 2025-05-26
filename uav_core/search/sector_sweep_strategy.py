# uav_core/search/sector_sweep_strategy.py

import math
import random
from uav_core.search.base_strategy import BaseSearchStrategy

class SectorSweepStrategy(BaseSearchStrategy):
    """
    Divide the search space into angular sectors based on the number of UAVs.
    Each UAV sweeps its assigned sector outward from the center, avoiding walls.
    """
    def __init__(self, map_size, uav_count, vision_range, wall_checker):
        super().__init__(map_size, uav_count, vision_range)
        self.wall_checker = wall_checker  # callable (x, y) -> True if wall
        self.center = (map_size[0] // 2, map_size[1] // 2)
        self.radius = min(map_size) // 2

    def assign_initial_positions(self):
        """Place UAVs near the center, slightly spaced."""
        cx, cy = self.center
        spacing = 5
        return [(cx + i * spacing, cy) for i in range(-self.uav_count // 2, self.uav_count // 2 + 1)][:self.uav_count]

    def next_move(self, uav_state, seen_areas):
        """
        Simple angular sector expansion logic.
        Each UAV gets a sector based on index and sweeps outward avoiding walls.
        """
        idx = uav_state['id']
        curr_x, curr_y = uav_state['pos']
        angle_step = 360 / self.uav_count
        angle = idx * angle_step

        step_size = self.vision_range // 2
        new_radius = uav_state.get('step', 1) * step_size
        uav_state['step'] = uav_state.get('step', 1) + 1

        dx = int(new_radius * math.cos(math.radians(angle)))
        dy = int(new_radius * math.sin(math.radians(angle)))
        new_x = self.center[0] + dx
        new_y = self.center[1] + dy

        if (0 <= new_x < self.map_size[0] and
            0 <= new_y < self.map_size[1] and
            not self.wall_checker(new_x, new_y)):
            return (new_x, new_y)
        else:
            # Try small random perturbations if blocked
            for _ in range(10):
                jitter_x = random.randint(-step_size, step_size)
                jitter_y = random.randint(-step_size, step_size)
                alt_x = curr_x + jitter_x
                alt_y = curr_y + jitter_y
                if (0 <= alt_x < self.map_size[0] and
                    0 <= alt_y < self.map_size[1] and
                    not self.wall_checker(alt_x, alt_y)):
                    return (alt_x, alt_y)

            return (curr_x, curr_y)  # fallback
