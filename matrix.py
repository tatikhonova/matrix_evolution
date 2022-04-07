import numpy as np


class MatrixIndivid:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness_value = None


if __name__ == '__main__':
    source_matrix = np.random.rand(10, 10)
    individ = MatrixIndivid(genotype=source_matrix)
