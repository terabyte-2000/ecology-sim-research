from __future__ import annotations

from neat import Genome, PlayerSave, World

import numpy as np
import pygame

import random

from collections.abc import Iterable


pygame.init()

KERNEL_SIZE = 3

assert KERNEL_SIZE % 2 == 1

WIDTH = 150
HEIGHT = 90
full_board = np.zeros((HEIGHT, WIDTH))

FOOD = 2
CREATURE = 1
SPACE = 0

IDEAL_FOOD_AMT = 100

BLACK = 0, 0, 0
WHITE = 255, 255, 255
RED = 255, 0, 0
YELLOW = 255, 255, 20

color_lookup: dict[int, tuple[int, int, int]] = {
    SPACE: BLACK, CREATURE: RED, FOOD: YELLOW}

POSSIBLE_MOVES = [(di, dj) for di in (-1, 0, 1)
                  for dj in (-1, 0, 1) if di != 0 or dj != 0]
# POSSIBLE_MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

font_size = 36


font = pygame.font.Font(None, font_size)


class Creature:
    """Base class for a player object. Many functions simply pass instead of doing stuff."""

    # No slots because with so many attributes it slows down significantly

    def __init__(self, x, y) -> None:
        """Initialize Creature."""
        self.x = x
        self.y = y
        full_board[y, x] = CREATURE

        self.food = 0

        self.fitness = 0.0
        self.unadjusted_fitness = 0
        self.best_score = 0.0  # Stores the score achieved used for replay
        self.dead = False
        self.score = 0.0
        self.gen = 0

        self.genome_inputs = KERNEL_SIZE * KERNEL_SIZE
        # each possible direction to go: (-1,-1), (-1, 0), etc., (1,1)
        self.genome_outputs = 8
        self.brain: Genome = self.start()

    def start(self) -> Genome:
        """Return new brain."""
        return Genome(self.genome_inputs, self.genome_outputs)

    def look(self) -> Iterable[float]:
        """Return inputs for the neural network."""
        # assuming odd kernel size
        dist_to_edge = (KERNEL_SIZE-1)//2

        return full_board[self.y].take(np.arange(self.x - dist_to_edge, self.x + dist_to_edge + 1), mode='wrap').flatten()

    def think(self, inputs: Iterable[float]) -> Iterable[float]:
        """Return decision from neural network."""
        return self.brain.feed_forward(inputs)

    def update(self, decision: Iterable[float]) -> None:
        """Move the player according to the outputs from the neural network."""
        # print(decision)
        # do = decision.index(max(self.decision))
        decision = np.array(decision)
        delta_y, delta_x = POSSIBLE_MOVES[np.argmax(decision)]

        full_board[self.y, self.x] = SPACE
        self.x += delta_x
        self.y += delta_y
        self.x %= WIDTH
        self.y %= HEIGHT
        if full_board[self.y, self.x] == FOOD:
            self.food += 1
        full_board[self.y, self.x] = CREATURE

        return

    def clone(self) -> Creature:
        """Return a clone of self."""
        clone = self.__class__(self.x, self.y)
        clone.brain = self.brain.clone()
        clone.fitness = float(self.fitness)
        clone.brain.generate_network()
        clone.gen = int(self.gen)
        clone.best_score = float(self.score)
        return clone

    def calculate_fitness(self) -> float:
        """Calculate the fitness of the AI."""
        return self.food

    def crossover(self, parent: Creature) -> Creature:
        """Return a Creature object by crossing over our brain and parent2's brain."""
        child = self.__class__(np.random.randint(WIDTH),
                               np.random.randint(HEIGHT))
        child.brain = self.brain.crossover(parent.brain)
        child.brain.generate_network()
        return child

    def save(self) -> PlayerSave:
        """Return a list containing important information about ourselves."""
        return (
            self.brain.save(),
            self.gen,
            self.dead,
            self.best_score,
            self.score,
        )

    # @classmethod
    # def load(cls, data: PlayerSave) -> Creature:
    #     """Return a BasePlayer Object with save data given."""
    #     self = cls()
    #     brain, self.gen, self.dead, self.best_score, self.score = data
    #     self.genome_inputs, self.genome_outputs = brain[:2]
    #     self.brain = Genome.load(brain)
    #     return self


for i in range(9):
    x = np.random.randint(0, HEIGHT-1)
    y = np.random.randint(0, WIDTH-1)
    for a in range(x-2, x+3):
        for b in range(y-2, y+3):
            if a >= 0 and a < HEIGHT and b >= 0 and b < WIDTH:
                full_board[a, b] = 2

original_board = full_board.copy()


def move_all(pop, neighborhood, steps=1):

    for _ in range(steps):
        pop_clone = np.array([creat.clone() for creat in pop])
        for i, creat in enumerate(pop_clone):
            pop[i].update(pop[i].think(pop[i].look()))


def generate_pop(n):
    arr = []
    for _ in range(n):
        i, j = np.random.randint(HEIGHT), np.random.randint(WIDTH)
        creat = Creature(j, i)
        arr.append(creat)
    return np.array(arr)


population = generate_pop(50)


FPS = 60
fpsClock = pygame.time.Clock()

CELL_SIZE = 7
SC_WIDTH, SC_HEIGHT = WIDTH*CELL_SIZE, HEIGHT*CELL_SIZE
screen = pygame.display.set_mode((SC_WIDTH, SC_HEIGHT))

done = False

cnt = 0
moves = 200

generation = 1

while not done:
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            pygame.quit()
            break

    if done:
        break

    cnt = (cnt + 1) % moves

    # Render the text
    text_render = font.render(f'Generation: {generation}', True, WHITE)

    # Calculate the position to place the text (top center)
    text_position = text_render.get_rect(center=(SC_WIDTH // 2, 50))

    # full_board[np.random.choice(np.ravel(np.where(full_board == 0)))] = 2
    # zeros = None

    amt_of_food = np.count_nonzero(full_board == FOOD)

    add_food = IDEAL_FOOD_AMT - amt_of_food > np.random.randint(-20, 20)

    # if add_food:
    if add_food:
        zeros = np.where(full_board == SPACE)

        r = np.random.randint(len(zeros[0]))
        z1 = zeros[0][r]
        z2 = zeros[1][r]

        full_board[z1, z2] = 2

    move_all(population, full_board, 1)

    if cnt == moves-1:
        # new generation
        generation += 1
        # generation.append(generation[-1]+1)
        fitnesses = [creat.calculate_fitness() for creat in population]

        # avg_fitness.append(sum(fitnesses)/len(fitnesses))
        # best_fitness.append(max(fitnesses))

        population = population[np.argsort(
            -np.array(fitnesses))]

        pop_copy = np.array([creat.clone() for creat in population])

        # for creat in population:
        #     full_board[creat.y, creat.x] = SPACE
        full_board = original_board.copy()

        for i, _ in enumerate(population):
            parent1 = pop_copy[min(
                int(np.random.exponential(5)), len(population)-1)]
            parent2 = pop_copy[min(
                int(np.random.exponential(5)), len(population)-1)]
            population[i] = parent1.crossover(parent2)

            # if np.random.randint(0, 10) == 1:
            #     population[i] = population[i].mutate()

        # np.random.choice(np.where(full_board == 0), size=5, replace=False)

    # black for 0, white for 1
    for y in range(0, SC_HEIGHT, CELL_SIZE):
        for x in range(0, SC_WIDTH, CELL_SIZE):
            col = color_lookup[int(
                full_board[y // CELL_SIZE, x // CELL_SIZE])]
            pygame.draw.rect(screen, col, (x, y, CELL_SIZE, CELL_SIZE))
    # Draw.

    screen.blit(text_render, text_position)

    pygame.display.flip()

    fpsClock.tick(FPS)
