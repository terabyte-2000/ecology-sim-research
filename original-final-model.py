import numpy as np
import pygame


def uniform_complex_around_unit_disc(shape):
    return np.sqrt(np.random.uniform(0, 1, shape)) * np.exp(1.j * np.random.uniform(0, 2 * np.pi, shape))


kernel_size = 5

assert kernel_size % 2 == 1

width = 100
height = 100
full_board = np.zeros((height, width))

FOOD = 2
CREATURE = 1
SPACE = 0

IDEAL_FOOD_AMT = 100

black = 0, 0, 0
white = 255, 255, 255
red = 255, 0, 0
yellow = 255, 255, 20

color_lookup = {SPACE: black, CREATURE: red, FOOD: yellow}

pygame.init()

font_size = 36

# Get the system font
# Passing None will use the default system font
font = pygame.font.Font(None, font_size)


# for _ in range(500):
#     zeros = np.where(full_board == SPACE)
#     r = np.random.randint(len(zeros[0]))
#     z1 = zeros[0][r]
#     z2 = zeros[1][r]
#     full_board[z1, z2] = 2

for i in range(5):
    x = np.random.randint(0, width-1)
    y = np.random.randint(0, height-1)
    for a in range(x-2, x+3):
        for b in range(y-2, y+3):
            if a >= 0 and a < width and b >= 0 and b < height:
                full_board[a, b] = 2


class Creature:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        full_board[y, x] = CREATURE
        self.food = 0
        # self.kernel = uniform_complex_around_unit_disc(
        #     (kernel_size, kernel_size))
        self.kernel = np.random.uniform(-3, 3, size=(kernel_size, kernel_size))

    def clone(self):
        creat = Creature(self.x, self.y)
        creat.food = self.food
        creat.kernel = np.copy(self.kernel)
        return creat

    def remove(self):
        full_board[self.y, self.x] = SPACE

    def update_pos(self, delta_x, delta_y):
        full_board[self.y, self.x] = SPACE
        self.x += delta_x
        self.y += delta_y
        self.x %= width
        self.y %= height
        if full_board[self.y, self.x] == FOOD:
            self.food += 1
        full_board[self.y, self.x] = CREATURE

    def eval_kernel(self, _neighborhood):
        return np.sum(self.kernel * _neighborhood)

    def all_neighborhoods(self, space):
        neighborhoods = []
        for del_i in -1, 0, 1:
            for del_j in -1, 0, 1:
                if del_i == 0 and del_j == 0:
                    continue

                i = (self.y + del_i) % height
                j = (self.x + del_j) % width

                # assuming odd kernel size
                dist_to_edge = (kernel_size-1)//2
                # _neighborhood = space[i-dist_to_edge:i +
                #                       dist_to_edge+1, j-dist_to_edge:j+dist_to_edge+1]

                _neighborhood = []
                for ii in range(i-dist_to_edge, i+dist_to_edge+1):
                    _neighborhood.append(space[ii % height].take(
                        np.arange(j-dist_to_edge, j+dist_to_edge+1), mode='wrap'))

                _neighborhood = np.array(_neighborhood)

                neighborhoods.append(_neighborhood)

        return np.array(neighborhoods)

    def best_move(self, space):
        deltas = [(del_i, del_j) for del_i in (-1, 0, 1)
                  for del_j in (-1, 0, 1) if del_i != 0 or del_j != 0]
        evaluated = []
        for _neighborhood in self.all_neighborhoods(space):
            evaled = self.eval_kernel(_neighborhood)
            # print(evaled)
            evaluated.append(evaled)

        return deltas[np.argmax(np.array(evaluated))]


def move_all(pop, neighborhood, steps=1):

    for _ in range(steps):
        neighborhood_clone = np.copy(neighborhood)
        pop_clone = np.array([creat.clone() for creat in pop])

        # print(pop_clone)
        # alive_pop = []
        for i, creat in enumerate(pop_clone):
            pop[i].update_pos(*creat.best_move(neighborhood_clone))
            # pop[i].food -= 0.2

            # if pop[i].food >= 0:
            #     alive_pop.append(pop[i])
            # else:
            #     pop[i].remove()

        # pop = np.copy(alive_pop)


# c1 = Creature(0, 0)
# full_board[0, 0] = 1
# print(c1.best_move(full_board))


def fitness(creat):
    return creat.food


def generate_pop(n):
    arr = []
    for _ in range(n):
        i, j = np.random.randint(height), np.random.randint(width)
        creat = Creature(j, i)
        arr.append(creat)
    return np.array(arr)


def crossover(creat1, creat2):
    i, j = np.random.randint(height), np.random.randint(width)
    kern = 0.5*(creat1.kernel + creat2.kernel)
    creat = Creature(j, i)
    creat.kernel = kern
    return creat


def mutate(creat):
    # return [indiv[0] + random.normalvariate(0, 2), indiv[1] + random.normalvariate(0, 2)]
    new_creat = Creature(creat.x, creat.y)
    # new_creat.kernel = creat.kernel + 0.5*uniform_complex_around_unit_disc((kernel_size, kernel_size))
    new_creat.kernel = creat.kernel + \
        np.random.uniform(-2, 2, size=(kernel_size, kernel_size))
    return new_creat

# print(full_board)
# move_all(population, full_board, 1)
# print('\n\n\n')
# print(full_board)


population = generate_pop(50)


# for i in range()


fps = 60
fpsClock = pygame.time.Clock()

cell_size = 7
sc_width, sc_height = width*cell_size, height*cell_size
screen = pygame.display.set_mode((sc_width, sc_height))

done = False

cnt = 0
moves = 200

generation = 1

while not done:
    screen.fill(black)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            pygame.quit()
            break

    if done:
        break

    cnt = (cnt + 1) % moves

    # Render the text
    text_render = font.render(f'Generation: {generation}', True, white)

    # Calculate the position to place the text (top center)
    text_position = text_render.get_rect(center=(sc_width // 2, 50))

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
        fitnesses = [fitness(creat) for creat in population]

        # avg_fitness.append(sum(fitnesses)/len(fitnesses))
        # best_fitness.append(max(fitnesses))

        population = population[np.argsort(
            -np.array(fitnesses))]

        pop_copy = np.array([creat.clone() for creat in population])

        for creat in population:
            full_board[creat.y, creat.x] = SPACE

        for i, _ in enumerate(population):
            parent1 = pop_copy[min(
                int(np.random.exponential(5)), len(population)-1)]
            parent2 = pop_copy[min(
                int(np.random.exponential(5)), len(population)-1)]
            population[i] = crossover(parent1, parent2)

            if np.random.randint(0, 10) == 1:
                population[i] = mutate(population[i])

        # np.random.choice(np.where(full_board == 0), size=5, replace=False)

    # black for 0, white for 1
    for y in range(0, sc_height, cell_size):
        for x in range(0, sc_width, cell_size):
            col = color_lookup[int(
                full_board[y // cell_size, x // cell_size])]
            pygame.draw.rect(screen, col, (x, y, cell_size, cell_size))
    # Draw.
    screen.blit(text_render, text_position)

    pygame.display.flip()

    fpsClock.tick(fps)
