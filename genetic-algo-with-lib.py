import numpy as np
from geneticalgorithm import geneticalgorithm as ga


def f(X):
    return np.sum(X)


varbound = np.array([[0, 10]]*5)

model = ga(function=f, dimension=5,
           variable_boundaries=varbound, variable_type='real')
model.run()

convergence = model.report
solution = model.output_dict

print(solution)
