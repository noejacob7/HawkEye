"""
Spiral search strategy for UAVs: spiral outward from center.
"""
import math
from uav_core.search.base_strategy import BaseSearchStrategy

class SpiralStrategy(BaseSearchStrategy):
    def __init__(self, map_size, uav_count, vision_range, wall_checker=None):
        super().__init__(map_size, uav_count, vision_range)
        self.center = (map_size[0] // 2, map_size[1] // 2)
        self.wall_checker = wall_checker

    def assign_initial_positions(self):
        # Start all UAVs at center
        return [self.center for _ in range(self.uav_count)]

    def next_move(self, uav_state, seen_areas):
        # TODO: implement spiral outward logic
        # Placeholder: stay in place
        return uav_state['pos']
