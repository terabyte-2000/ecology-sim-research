import pygame
import numpy as np
# from pygame.locals import *

pygame.init()

fps = 60
fpsClock = pygame.time.Clock()

width, height = 640, 480
screen = pygame.display.set_mode((width, height))

black = 0, 0, 0
white = 255, 255, 255

# Game loop.

state = np.random.randint(0, 2, (height//2, width//2))


def update_state(st):
    _st = np.random.randint(0, 2, (height//2, width//2))
    np.copyto(st, _st)


done = False

while not done:
    screen.fill((100, 100, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            pygame.quit()
            break

    if not done:
        update_state(state)

        # black for 0, white for 1
        for y in range(0, height, 2):
            for x in range(0, width, 2):
                col = black if state[y // 2, x // 2] == 0 else white
                pygame.draw.rect(screen, col, (x, y, 2, 2))
        # Draw.
        pygame.display.flip()

        fpsClock.tick(fps)
