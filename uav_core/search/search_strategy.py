# uav_core/search/search_strategy.py

from abc import ABC, abstractmethod

class BaseSearchStrategy(ABC):
    """
    Abstract base class for UAV search strategies.
    Each strategy must implement how a UAV should plan its next move.
    """
    def __init__(self, map_size, uav_count, vision_range):
        self.map_size = map_size
        self.uav_count = uav_count
        self.vision_range = vision_range

    @abstractmethod
    def assign_initial_positions(self):
        """
        Return initial positions for each UAV.
        """
        pass

    @abstractmethod
    def next_move(self, uav_state, seen_areas):
        """
        Given a UAV's current state and seen areas, return next move.
        """
        pass
