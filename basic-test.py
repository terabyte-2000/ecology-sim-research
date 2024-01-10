import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

ROWS = 7
COLS = 7
MAX_FOOD = 5

food_distr = np.random.randint(0, MAX_FOOD+1, size=(ROWS, COLS))


class Agent:
    maxId = 0

    def __init__(self, pos: tuple[int, int]):
        self.food = MAX_FOOD
        self.row: int = pos[0]
        self.col: int = pos[1]
        self.alive = True
        Agent.maxId += 1
        self.id = Agent.maxId

    def __str__(self) -> str:
        return f'''
        ID: {self.id}
        Food: {self.food}
        Position: {self.row}, {self.col}
        Alive: {self.alive}
        '''

    def step(self):

        if self.food <= 0:
            self.alive = False
            return

        self.food -= 2

        self.row += np.random.choice((-1, 0, 1))
        self.row = np.clip(self.row, 0, ROWS-1)
        # while self.row >= ROWS or self.row < 0:
        #     self.row += np.random.choice((-1, 1))

        self.col += np.random.choice((-1, 0, 1))
        self.col = np.clip(self.col, 0, COLS-1)
        # while self.col >= COLS or self.col < 0:
        #     self.col += np.random.choice((-1, 1))

        food_at_pt: int = food_distr[self.row, self.col]
        if food_at_pt != 0:
            max_amt = np.min([food_at_pt, MAX_FOOD - self.food])
            amt = np.random.randint(0,  max_amt+1)
            food_distr[self.row, self.col] -= amt
            self.food += amt


print(food_distr)

a1 = Agent((0, 0))
a2 = Agent((ROWS-1, COLS-1))
a3 = Agent((3, 3))
a4 = Agent((2, 4))
a5 = Agent((1, 3))

agents = [a1, a2, a3, a4, a5]

pallete = sns.color_palette("viridis", as_cmap=True)

fig, ax = plt.subplots(ncols=6, sharey=True)

distrs = [food_distr.copy()]

# sns.heatmap(food_distr.copy(), linewidth=0.5, ax=ax[0], cmap=pallete)
for i in range(5):
    for a in agents:
        a.step()
        print(a)

    print(food_distr)
    distrs.append(food_distr.copy())


for i, dist in enumerate(distrs):
    sns.heatmap(dist, linewidth=0.5, ax=ax[i], cmap=pallete)


plt.tight_layout()

plt.show()
