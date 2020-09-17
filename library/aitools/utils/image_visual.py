import pygame
import math
import os

os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"


class ImageVisual(object):
    def __init__(self, width, height, cellSize=10):
        self.width = width
        self.height = height
        self.cellSize = cellSize

        pygame.init()

    def close(self):
        pygame.quit()

    def display(self, images, *, rowsize=None):
        realRowsize = len(images)
        realColsize = 1
        if rowsize is not None:
            realRowsize = min(realRowsize, rowsize)
            realColsize = math.floor(len(images) / realRowsize)

        displayWidth = (self.width + 1) * realRowsize + 1
        displayHeight = (self.height + 1) * realColsize + 1

        display = pygame.display.set_mode(
            (self.cellSize * displayWidth, self.cellSize * displayHeight)
        )
        display.fill((30, 200, 30,))
        for i, image in enumerate(images):
            self._image(display, (i % realRowsize, math.floor(i / realRowsize)), image)
        pygame.display.update()

    def _image(self, display, coordinates, data):
        for pixel, intensity in enumerate(data):
            self._cell(display, coordinates, pixel, 255 * float(intensity))

    def _cell(self, display, coordinates, pixel, intensity):
        color = tuple([int(intensity) for _ in range(3)])

        x = int(pixel % self.width)
        y = int((pixel - x) / self.width)

        offsetX = coordinates[0] * (self.width + 1) + 1
        offsetY = coordinates[1] * (self.height + 1) + 1
        pygame.draw.rect(
            display,
            color,
            (
                (x + offsetX) * self.cellSize,
                (y + offsetY) * self.cellSize,
                self.cellSize,
                self.cellSize,
            ),
        )
