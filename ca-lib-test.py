import random

from cellular_automaton import CellularAutomaton, MooreNeighborhood, CAWindow, EdgeRule

ALIVE = [1.0]
DEAD = [0]


class ConwaysCA(CellularAutomaton):
    """ Cellular automaton with the evolution rules of conways game of life """

    def __init__(self):
        super().__init__(dimension=[100, 100],
                         neighborhood=MooreNeighborhood(EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS))

    def init_cell_state(self, __):  # pylint: disable=no-self-use
        return random.choices([0, 1], weights=[14, 1])

    def evolve_rule(self, last_cell_state, neighbors_last_states):
        new_cell_state = last_cell_state
        alive_neighbours = self.__count_alive_neighbours(neighbors_last_states)
        if last_cell_state == DEAD and alive_neighbours == 3:
            new_cell_state = ALIVE
        if last_cell_state == ALIVE and alive_neighbours < 2:
            new_cell_state = DEAD
        if last_cell_state == ALIVE and 1 < alive_neighbours < 4:
            new_cell_state = ALIVE
        if last_cell_state == ALIVE and alive_neighbours > 3:
            new_cell_state = DEAD
        return new_cell_state

    @staticmethod
    def __count_alive_neighbours(neighbours):
        alive_neighbors = 0
        for n in neighbours:
            alive_neighbors += n[0]
        return alive_neighbors


def state_to_color(curr_state):
    return (0,)*3 if curr_state[0] == 0 else (255,)*3


if __name__ == "__main__":
    CAWindow(cellular_automaton=ConwaysCA(),
             window_size=(1000, 830),
             state_to_color_cb=state_to_color).run(evolutions_per_second=40)
