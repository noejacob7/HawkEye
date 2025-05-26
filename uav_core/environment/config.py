"""
uav_core.environment.config

Holds global configuration constants for UAV simulation.
"""

# Default UAV parameters
DEFAULT_FOV_ANGLE = 60       # degrees
DEFAULT_VISION_RANGE = 100   # units
DEFAULT_SPEED = 5            # units per tick

# Battery model parameters
BATTERY_CAPACITY = 100.0
DRAIN_RATE_MOVE = 0.05       # per unit distance
DRAIN_RATE_SCAN = 0.2        # per scan action
