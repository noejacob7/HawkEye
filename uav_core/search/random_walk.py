"""
Random walk search strategy: UAV chooses random valid move each step.
"""
import random
from uav_core.search.base_strategy import BaseSearchStrategy

class RandomWalkStrategy(BaseSearchStrategy):
    def __init__(self, map_size, uav_count, vision_range, wall_checker=None):
        super().__init__(map_size, uav_count, vision_range)
        self.wall_checker = wall_checker

    def assign_initial_positions(self):
        # Start at random positions
        return [(random.randint(0, self.map_size[0]-1), random.randint(0, self.map_size[1]-1))
                for _ in range(self.uav_count)]

    def next_move(self, uav_state, seen_areas):
        curr_x, curr_y = uav_state['pos']
        # choose a random direction
        dx = random.choice([-1, 0, 1]) * self.vision_range
        dy = random.choice([-1, 0, 1]) * self.vision_range
        new_x = max(0, min(self.map_size[0]-1, curr_x + dx))
        new_y = max(0, min(self.map_size[1]-1, curr_y + dy))
        if self.wall_checker and self.wall_checker(new_x, new_y):
            return (curr_x, curr_y)
        return (new_x, new_y)
