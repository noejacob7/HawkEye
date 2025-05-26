"""
uav_core.environment.collision_utils

Raycasting and wall detection utilities for field-of-view and path planning.
"""

def bresenham_line(x0, y0, x1, y1):
    """
    Generate points on a line between (x0, y0) and (x1, y1) using Bresenham's algorithm.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points

def has_line_of_sight(start, end, map_loader):
    """
    Check if straight path from start to end intersects any walls.
    start, end: (x,y)
    map_loader: MapLoader instance
    """
    for x, y in bresenham_line(int(start[0]), int(start[1]), int(end[0]), int(end[1])):
        if map_loader.is_wall(x, y):
            return False
    return True
