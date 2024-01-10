import random
from copy import deepcopy

def fitness(x):
    return (x+3)*(x+1)*(x-5)*(x-4)


def generate_pop(n, lo=-100, hi=100):
    return [random.uniform(lo, hi) for _ in range(n)]


population = generate_pop(100)

old_pop = deepcopy(population)
# print(population[0:15])


def crossover(p1, p2):
    return 0.5*(p1+p2)


def n_parent_crossover(parents):
    return sum(parents)/len(parents)


def mutate(indiv):
    return indiv + random.normalvariate(0, 2)


def overall_fitness(threshold):
    return sum(fitness(indiv) for indiv in population[0:threshold])


def first():
    for i in range(15):
        population.sort(key=fitness)
        pop_copy = deepcopy(population)

        for i, _ in enumerate(population):
            parent1 = pop_copy[int(random.expovariate(1))]
            parent2 = pop_copy[int(random.expovariate(1))]
            population[i] = crossover(parent1, parent2)

            if random.randint(0, 20) == 1:
                population[i] = mutate(population[i])


def second():
    for i in range(15):
        population.sort(key=fitness)

        parent1 = population[int(random.expovariate(1))]
        parent2 = population[int(random.expovariate(1))]

        for i, _ in enumerate(population):
            population[i] = crossover(parent1, parent2)

            if random.randint(0, 20) == 1:
                population[i] = mutate(population[i])


def third(n):
    for i in range(15):
        population.sort(key=fitness)

        parents = [population[int(random.expovariate(1))] for _ in range(n)]

        for i, _ in enumerate(population):
            population[i] = n_parent_crossover(parents)

            if random.randint(0, 20) == 1:
                population[i] = mutate(population[i])


def fitness_from_n_iters(func, iters):
    global population
    _old_pop = deepcopy(population)
    total = 0
    for _ in range(iters):
        func()
        total += overall_fitness(15)/iters
        population = deepcopy(_old_pop)
    return total


def winning_from_n_iters(iters):
    global population
    _old_pop = deepcopy(population)
    first_wins = 0
    second_wins = 0
    third_wins = 0
    wth = 0

    for _ in range(iters):
        # population = generate_pop(100)
        # _old_pop = deepcopy(population)

        first()
        first_score = overall_fitness(15)
        population = deepcopy(_old_pop)
        second()
        second_score = overall_fitness(15)
        population = deepcopy(_old_pop)
        third(4)
        third_score = overall_fitness(15)
        population = deepcopy(_old_pop)

        # ranked = [third_score, second_score, third_score]
        # ranked.sort()

        if first_score < second_score and first_score < third_score:
            first_wins += 1
        elif second_score < first_score and second_score < third_score:
            second_wins += 1
        elif third_score < first_score and third_score < second_score:
            third_wins += 1
        else:
            wth += 1

    return (first_wins, second_wins, third_wins, wth)


# # first()
# print(fitness_from_n_iters(first, 1000))
# print("First:")
# print(population[0:15])
# # print(overall_fitness(15))

# print("-"*50)
# print("Second:")

# population = [*old_pop]
# # second()
# print(fitness_from_n_iters(second, 1000))
# print(population[0:15])
# # print(overall_fitness(15))

# print("-"*50)
# print("Third:")
# population = [*old_pop]
# # third(4)
# print(fitness_from_n_iters(lambda: third(4), 1000))
# print(population[0:15])
# # print(overall_fitness(15))


print(winning_from_n_iters(500))
