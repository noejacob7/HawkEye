# components/drone.py
import pygame

class Drone:
    def __init__(self, name, x, y, image):
        self.name = name
        self.x = x
        self.y = y
        self.image = image
        self.font = pygame.font.SysFont("Arial", 16)

    def draw(self, surface):
        surface.blit(self.image, (self.x, self.y))
        label = self.font.render(self.name, True, (255, 255, 255))
        surface.blit(label, (self.x + 10, self.y - 20))
