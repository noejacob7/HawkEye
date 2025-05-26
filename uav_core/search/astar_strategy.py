"""
A* search strategy for UAVs, avoids walls using environment data.
"""
import heapq
from uav_core.search.base_strategy import BaseSearchStrategy

class AStarStrategy(BaseSearchStrategy):
    def __init__(self, map_size, uav_count, vision_range, map_loader):
        super().__init__(map_size, uav_count, vision_range)
        self.map_loader = map_loader
        self.center = (map_size[0]//2, map_size[1]//2)

    def assign_initial_positions(self):
        # Start UAVs at center
        return [self.center for _ in range(self.uav_count)]

    def heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def next_move(self, uav_state, seen_areas):
        # Placeholder: always return current position
        return uav_state['pos']
