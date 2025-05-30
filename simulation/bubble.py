# components/bubble.py
import pygame

class Bubble:
    def __init__(self, text, position, font, color=(0, 0, 0), background=(255, 255, 255)):
        self.text = text
        self.position = position
        self.font = font
        self.color = color
        self.background = background
        self.padding = 10
        self.visible = True

    def draw(self, screen):
        if not self.visible:
            return

        rendered_text = self.font.render(self.text, True, self.color)
        text_rect = rendered_text.get_rect()
        bubble_rect = pygame.Rect(
            self.position[0], self.position[1],
            text_rect.width + 2 * self.padding,
            text_rect.height + 2 * self.padding
        )

        pygame.draw.rect(screen, self.background, bubble_rect, border_radius=10)
        pygame.draw.rect(screen, self.color, bubble_rect, 2, border_radius=10)
        screen.blit(rendered_text, (self.position[0] + self.padding, self.position[1] + self.padding))

    def set_text(self, new_text):
        self.text = new_text

    def hide(self):
        self.visible = False

    def show(self):
        self.visible = True
