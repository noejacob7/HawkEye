"""
uav_core.energy.battery_model

Simple battery model for UAVs.
"""

class BatteryModel:
    def __init__(self, capacity=100.0, drain_rate_move=0.05, drain_rate_scan=0.2):
        self.capacity = capacity
        self.level = capacity
        self.drain_rate_move = drain_rate_move
        self.drain_rate_scan = drain_rate_scan

    def drain_movement(self, distance):
        """Drain battery based on movement distance."""
        self.level -= self.drain_rate_move * distance
        if self.level < 0:
            self.level = 0

    def drain_scan(self):
        """Drain battery for scanning action."""
        self.level -= self.drain_rate_scan
        if self.level < 0:
            self.level = 0

    def is_depleted(self):
        return self.level <= 0

    def get_level(self):
        return self.level
