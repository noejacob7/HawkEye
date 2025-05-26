"""
uav_core.controller.uav

Defines the UAV agent class.
"""

class UAV:
    """
    UAV agent with position, battery, memory, and search strategy.
    """
    def __init__(self, id, position, battery_model, search_strategy, memory, comm_bus):
        self.id = id
        self.position = position  # tuple (x, y, z) or (x, y)
        self.battery = battery_model
        self.strategy = search_strategy
        self.memory = memory
        self.comm_bus = comm_bus
        self.state = {'step': 1, 'pos': position, 'id': id}

    def move(self, new_position):
        """Update position and drain battery by movement cost."""
        self.position = new_position
        self.state['pos'] = new_position
        # battery consumption handled externally

    def capture_image(self):
        """Simulate image capture and store in local memory."""
        # Placeholder: returns dummy image data
        return None

    def send_to_peer(self, data):
        """Send data to other UAVs via message bus."""
        self.comm_bus.broadcast(self.id, data)

    def receive_data(self, sender_id, data):
        """Receive data from peers and update local memory."""
        # Placeholder: integrate received data
        pass
