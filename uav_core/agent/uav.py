# uav_core/agent/uav.py

import math
import uuid
import random

class UAV:
    def __init__(self, start_pos, fov_angle=60, vision_range=100, battery_capacity=100):
        self.id = str(uuid.uuid4())[:8]  # Short unique identifier
        self.position = start_pos         # (x, y) tuple
        self.facing_angle = random.uniform(0, 360)  # Degrees
        self.fov_angle = fov_angle        # Degrees
        self.vision_range = vision_range  # Units (pixels or meters)
        self.battery = battery_capacity   # Percentage
        self.log = []                    # Log of actions
        self.memory = []                 # Stores image evidence or detections
        self.detected_targets = set()    # Avoid re-scanning same object

    def move_towards(self, target_pos, speed=5):
        if self.battery <= 0:
            return

        dx = target_pos[0] - self.position[0]
        dy = target_pos[1] - self.position[1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            return

        ratio = min(speed / dist, 1.0)
        new_x = self.position[0] + dx * ratio
        new_y = self.position[1] + dy * ratio
        self.position = (new_x, new_y)

        self.battery -= 0.05 * ratio * speed  # Cost of movement
        self.log.append(f"Moved to {self.position}, battery: {self.battery:.1f}%")

    def rotate_towards(self, target_pos):
        dx = target_pos[0] - self.position[0]
        dy = target_pos[1] - self.position[1]
        angle_rad = math.atan2(dy, dx)
        self.facing_angle = math.degrees(angle_rad)
        self.log.append(f"Rotated to angle {self.facing_angle:.1f}")

    def in_fov(self, obj_pos):
        dx = obj_pos[0] - self.position[0]
        dy = obj_pos[1] - self.position[1]
        angle_to_obj = math.degrees(math.atan2(dy, dx)) % 360
        angle_diff = (angle_to_obj - self.facing_angle + 360) % 360
        return angle_diff <= self.fov_angle / 2 or angle_diff >= 360 - self.fov_angle / 2

    def scan_object(self, obj_id, obj_pos):
        if self.battery <= 0 or obj_id in self.detected_targets:
            return None

        if self.in_fov(obj_pos) and self.distance_to(obj_pos) <= self.vision_range:
            self.detected_targets.add(obj_id)
            self.memory.append((obj_id, obj_pos))
            self.battery -= 0.2  # Cost of scanning
            self.log.append(f"Scanned object {obj_id} at {obj_pos}")
            return {
                "uav_id": self.id,
                "object_id": obj_id,
                "position": obj_pos,
                "evidence": f"img_{obj_id}_{self.id}.jpg"
            }
        return None

    def distance_to(self, pos):
        return math.hypot(pos[0] - self.position[0], pos[1] - self.position[1])

    def get_status(self):
        return {
            "id": self.id,
            "position": self.position,
            "battery": round(self.battery, 2),
            "angle": round(self.facing_angle, 2),
            "scanned": list(self.detected_targets)
        }
