# components/table_display.py
import pygame
import pandas as pd

FONT_SIZE = 18
TABLE_WIDTH = 250
TABLE_HEIGHT = 100
PADDING = 5

class TableDisplay:
    def __init__(self, x, y, font_path=None):
        self.x = x
        self.y = y
        self.width = TABLE_WIDTH
        self.height = TABLE_HEIGHT
        self.bg_color = (240, 240, 240)
        self.border_color = (0, 0, 0)
        self.text_color = (0, 0, 0)
        self.font = pygame.font.Font(font_path, FONT_SIZE)

    def render(self, screen, df: pd.DataFrame, title="Top-K Matches"):
        # Draw background
        table_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, self.bg_color, table_rect)
        pygame.draw.rect(screen, self.border_color, table_rect, 2)

        # Title
        title_surf = self.font.render(title, True, self.text_color)
        screen.blit(title_surf, (self.x + PADDING, self.y + PADDING))

        # Table rows
        row_y = self.y + PADDING + FONT_SIZE
        for idx, (_, row) in enumerate(df.iterrows()):
            text = f"{row['Rank']}. {row['Video']} ({row['CosSim']:.2f})"
            row_surf = self.font.render(text, True, self.text_color)
            screen.blit(row_surf, (self.x + PADDING, row_y))
            row_y += FONT_SIZE + 2
