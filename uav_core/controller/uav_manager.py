"""
UAV Manager: Spawns UAVs, steps through simulation ticks, updates states.
"""

from uav_core.controller.uav import UAV

class UAVManager:
    def __init__(self, map_size, strategy_factory, uav_count, battery_model_factory, memory_factory, comm_bus, vision_range=None):
        # Initialize map, UAV count, and factories
        self.map_size = map_size
        self.uav_count = uav_count
        self.vision_range = vision_range
        # Strategy factory should accept (map_size, uav_count, vision_range, ...)
        self.strategy = strategy_factory(map_size, uav_count, vision_range)
        self.battery_model_factory = battery_model_factory
        self.memory_factory = memory_factory
        self.comm_bus = comm_bus
        self.uavs = []

    def spawn_uavs(self):
        """Instantiate UAV agents with initial positions and resources."""
        initial_positions = self.strategy.assign_initial_positions()
        self.uavs = []
        for idx, pos in enumerate(initial_positions):
            battery = self.battery_model_factory()
            memory = self.memory_factory()
            uav = UAV(id=idx, position=pos, battery_model=battery,
                      search_strategy=self.strategy, memory=memory, comm_bus=self.comm_bus)
            self.uavs.append(uav)
        return self.uavs

    def step(self, seen_areas):
        """Advance simulation by one tick: move all UAVs according to strategy."""
        moves = []
        for uav in self.uavs:
            new_pos = uav.strategy.next_move(uav.state, seen_areas)
            uav.move(new_pos)
            moves.append((uav.id, new_pos))
        return moves
