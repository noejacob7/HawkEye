"""
uav_core.environment.map_loader

Loads and provides access to map data (walls, free zones).
"""

class MapLoader:
    """
    MapLoader holds a 2D grid of map cells: 0 free, 1 wall.
    """
    def __init__(self, grid):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

    def is_wall(self, x, y):
        """Return True if cell (x, y) is a wall or out of bounds."""
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True
        return self.grid[y][x] == 1
