import pygame

def load_image(path, scale=None):
    image = pygame.image.load(path).convert_alpha()
    if scale:
        image = pygame.transform.scale(image, scale)
    return image

def draw_uav_state(screen, x, y, drone_img, cloud_img, uav):
    screen.blit(drone_img, (x, y))
    screen.blit(cloud_img, (x + 80, y - 40))
    font = pygame.font.SysFont("Arial", 14)
    for i, line in enumerate(uav["log"][-3:]):
        text = font.render(line, True, (0, 0, 0))
        screen.blit(text, (x + 90, y - 30 + i * 15))

def draw_arrow(screen, start, end, arrow_img):
    mid_x = (start[0] + end[0]) // 2
    mid_y = (start[1] + end[1]) // 2
    screen.blit(arrow_img, (mid_x, mid_y))
