import numpy as np
from matrix import MatrixIndivid
from functools import partial
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from operator import attrgetter
import math

def matrix_from_source(source_matrix, matrix_size, bound_value=0.3):
    size_a, size_b = matrix_size
    matrix = bound_value * np.random.uniform(-1, 1, size=(size_a, size_b)) + source_matrix
    return matrix

def initial_population(pop_size, source_matrix, bound_value=0.2):
    size = source_matrix.shape

    pop = []
    for _ in range(pop_size):
        u = matrix_from_source(source_matrix, size, bound_value=bound_value)
        pop.append(MatrixIndivid(genotype=u))
    return pop

def evolution(source_matrix, crossover, fitness):
    mutation = partial(mutation_gauss, mu=0, sigma=0.3, prob_global=0.05)
    init_population = partial(initial_population, source_matrix=source_matrix, bound_value=0.5)
    evo_operators = {'fitness': fitness,
                     'parent_selection': partial(select_by_tournament, tournament_size=20),
                     'mutation': partial(mutated_individ_only_u, mutate=mutation),
                     'crossover': partial(separate_crossover_only_u, crossover=crossover),
                     'initial_population': init_population}
    meta_params = {'pop_size': 100, 'generations': 500, 'bound_value': 0.5,
                   'selection_rate': 0.2, 'crossover_rate': 0.60, 'random_selection_rate': 0.2, 'mutation_rate': 0.1}

    return evo_operators, meta_params

def k_point_crossover(parent_first, parent_second, type='horizontal', k=3, **kwargs):
    size = parent_first.shape
    child_first, child_second = np.zeros(size), np.zeros(size)

    if type == 'random':
        type = np.random.choice(['horizontal', 'vertical'])

    if type == 'horizontal':
        points = __random_cross_points(max_size=size[0], k=k)
        parents = [parent_first, parent_second]
        parent_idx = 0

        for point_idx in range(1, len(points)):
            point_from, point_to = points[point_idx - 1], points[point_idx]

            child_first[point_from: point_to] = parents[parent_idx][point_from:point_to]
            child_second[point_from:point_to] = parents[(parent_idx + 1) % 2][point_from:point_to]

            parent_idx = (parent_idx + 1) % 2
    elif type == 'vertical':
        points = __random_cross_points(max_size=size[0], k=k)

        parents = [parent_first, parent_second]
        parent_idx = 0

        for point_idx in range(1, len(points)):
            point_from, point_to = points[point_idx - 1], points[point_idx]

            child_first[:, point_from: point_to] = parents[parent_idx][:, point_from:point_to]
            child_second[:, point_from:point_to] = parents[(parent_idx + 1) % 2][:, point_from:point_to]

            parent_idx = (parent_idx + 1) % 2

    return child_first, child_second


def geo_crossover(parent_first, parent_second, random_box=True, **kwargs):
    size = parent_first.shape

    if random_box:
        top_left = (np.random.randint(low=0, high=size[0]),
                    np.random.randint(low=0, high=size[0]))
        box_size = np.random.randint(low=0, high=size[0])
        bottom_right = (top_left[0] + box_size, top_left[1] + box_size)
    else:
        box_size = kwargs['box_size']
        top_left = kwargs['top_left']
        bottom_right = (top_left[0] + box_size, top_left[1] + box_size)

    inside_mask = np.zeros(size)
    inside_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 1.0

    child_first = inside_mask * parent_first + (1.0 - inside_mask) * parent_second
    child_second = inside_mask * parent_second + (1.0 - inside_mask) * parent_first

    return child_first, child_second

def geo_crossover_fixed_box(parent_first, parent_second, box_size, **kwargs):
    size = parent_first.shape
    top_left = np.random.randint(low=0, high=size)

    child_first, child_second = geo_crossover(parent_first=parent_first, parent_second=parent_second, random_box=False,
                                              top_left=top_left, box_size=box_size)

    return child_first, child_second

def __random_cross_points(max_size, k=3):
    points = random.sample(range(0, max_size), k)
    if 0 not in points:
        points.append(0)
    if max_size not in points:
        points.append(max_size)
    points = sorted(points)
    return points

class BasicEvoStrategy:
    def __init__(self, evo_operators: dict, meta_params: dict, history, source_matrix, real_solution=None):
        self.fitness = evo_operators['fitness']
        self.select_parents = evo_operators['parent_selection']
        self.mutate = evo_operators['mutation']
        self.crossover = evo_operators['crossover']
        self.initial_population = evo_operators['initial_population']

        self.meta_params = meta_params
        self.pop_size = meta_params['pop_size']
        self.generations = meta_params['generations']
        self.pop = []
        self.cur_gen = -1
        self.source_matrix = source_matrix
        self.matrix_size = source_matrix.shape

        self.history = history
        self.__first_min_fitness = 10e9
        self.mae = []
        self.real_solution = real_solution

    def run(self):
        # self.history.init_new_run()
        self.__init_population()

        while not self.__stop_criteria():

            self.__assign_fitness_values()
            top = self.graded_by_fitness()[0]

            if self.cur_gen == 0:
                self.__first_min_fitness = top.fitness_value
            # self.__history_callback()

            offspring = self.__new_offspring()

            mutations_amount = int(len(offspring) * self.meta_params['mutation_rate'])
            for _ in range(mutations_amount):
                idx = np.random.randint(len(offspring) - 1)
                offspring[idx] = self.mutate(offspring[idx])
            self.pop = offspring
            self.cur_gen += 1

            # if self.cur_gen % 10 == 0:
            #     print(self.cur_gen)
            #     print(f'Best candidate with fitness: {top.fitness_value}')
            #     print(f'Best candidate with normed fitness: {normed_top}')
            #     print(f'Average fitness in population: {avg}')

                # top_matrix = top.genotype
                # self.evo_print_solution(np.array(top_matrix))
        top_matrix = top.genotype
        # self.evo_print_solution(np.array(top_matrix))

        if self.real_solution is not None:
            top_matrix = np.transpose(top_matrix)
            self.mae = np.mean(abs(top_matrix-self.real_solution))


    def __init_population(self):
        self.pop = self.initial_population(self.pop_size)
        self.cur_gen = 0

    def __assign_fitness_values(self):
        for individ in self.pop:
            u = individ.genotype
            individ.fitness_value = self.fitness(u)

    def evo_print_solution(self, matrix):
        x = np.linspace(0, 1, 11)
        t = np.linspace(0, 1, 11)

        grid = []
        grid.append(x)
        grid.append(t)

        grid = np.meshgrid(*grid)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(grid[0].reshape(-1), grid[1].reshape(-1), matrix.reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()

    def graded_by_fitness(self):
        self.__assign_fitness_values()
        graded = sorted(self.pop, key=attrgetter('fitness_value'))
        return graded

    def __new_offspring(self):
        offspring = []
        selected_amount = int(len(self.pop) * self.meta_params['selection_rate'])
        selected_parents = self.select_parents(candidates=self.pop, k=selected_amount)
        offspring.extend(selected_parents)

        childs_total = int(len(self.pop) * self.meta_params['crossover_rate'])
        childs_amount = 0

        while childs_amount < childs_total:
            parent_first, parent_second = np.random.choice(selected_parents), np.random.choice(selected_parents)

            child_first, child_second = self.crossover(parent_first, parent_second, current_gen=self.cur_gen)

            offspring.extend([child_first, child_second])
            childs_amount += 2
        random_chosen = self.__diversity(rate=self.meta_params['random_selection_rate'], fraction_worst=0.5)
        offspring.extend(random_chosen)
        return offspring

    def __survived(self, survive_rate=0.2):
        survived = select_k_best(candidates=self.pop, k=int(len(self.pop) * survive_rate))
        return survived

    def __diversity(self, rate=0.1, fraction_worst=0.5):
        k_worst = int((1.0 - fraction_worst) * len(self.pop))
        worst_candidates = self.graded_by_fitness()[k_worst:]
        random_chosen = np.random.choice(worst_candidates, int(len(self.pop) * rate))

        return random_chosen

    def __stop_criteria(self):
        return self.cur_gen >= self.generations

def select_k_best(candidates, k):
    assert k <= len(candidates)

    graded = sorted(candidates, key=attrgetter('fitness_value'))

    return graded[:k]

def mutation_gauss(candidate, mu, sigma, prob_global):
    source_shape = candidate.shape
    resulted = np.ndarray.flatten(candidate)

    chosen_values_amount = math.ceil(prob_global * len(resulted))
    idx_to_mutate = np.random.choice(np.arange(0, len(resulted)), chosen_values_amount, replace=False)
    for idx in idx_to_mutate:
        resulted[idx] = np.random.normal(mu, sigma)

    return resulted.reshape(source_shape)

def select_by_tournament(candidates, k, tournament_size=10):
    chosen = []
    for _ in range(k):
        aspirants = np.random.choice(candidates, tournament_size)
        chosen.append(min(aspirants, key=attrgetter('fitness_value')))

    return chosen

def mutated_individ_only_u(source_individ: MatrixIndivid, mutate):
    u = source_individ.genotype

    u_mutated = mutate(candidate=u)
    resulted = MatrixIndivid(genotype=u_mutated)

    return resulted

def separate_crossover_only_u(parent_first: MatrixIndivid, parent_second: MatrixIndivid, crossover, **kwargs):
    u_first, u_second = crossover(parent_first.genotype, parent_second.genotype, **kwargs)

    child_first = MatrixIndivid(genotype=u_first)
    child_second = MatrixIndivid(genotype=u_second)

    return child_first, child_second
