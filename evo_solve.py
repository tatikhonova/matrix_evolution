from solver.solver_matrix import op_dict_to_list, operator_unify, bnd_prepare_matrix, matrix_loss
import torch
import numpy as np
import time
from functools import partial
from MatrixEvolution.evo_algorithm_new import (BasicEvoStrategy,
evolution,
k_point_crossover,
geo_crossover_fixed_box
)
import time

device = torch.device('cpu')

x = torch.from_numpy(np.linspace(0, 1, 11))
t = torch.from_numpy(np.linspace(0, 1, 11))

grid = []
grid.append(x)
grid.append(t)

grid = np.meshgrid(*grid)
grid = torch.tensor(grid, device=device)

# Initial conditions at t=0
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float().to(device)

# u(0,x)=sin(pi*x)
bndval1 = torch.sin(np.pi * bnd1[:, 0]).to(device)

# Initial conditions at t=1
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([1], dtype=np.float64))).float().to(device)

# u(1,x)=sin(pi*x)
bndval2 = torch.sin(np.pi * bnd2[:, 0]).to(device)

# Boundary conditions at x=0
bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float().to(device)

# u(0,t)=0
bndval3 = torch.from_numpy(np.zeros(len(bnd3), dtype=np.float64)).to(device)

# Boundary conditions at x=1
bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float().to(device)

# u(1,t)=0
bndval4 = torch.from_numpy(np.zeros(len(bnd4), dtype=np.float64)).to(device)

# Putting all bconds together
bconds = [[bnd1, bndval1], [bnd2, bndval2], [bnd3, bndval3], [bnd4, bndval4]]

wave_eq = {
    '4*d2u/dx2**1':
        {
            'coeff': 4,
            'd2u/dx2': [0, 0],
            'pow': 1
        },
    '-d2u/dt2**1':
        {
            'coeff': -1,
            'd2u/dt2': [1,1],
            'pow':1
        }
}

operator = op_dict_to_list(wave_eq)
unified_operator = operator_unify(operator)
b_prepared = bnd_prepare_matrix(bconds, grid)


fitness_function = lambda x: matrix_loss(torch.tensor(x), grid, unified_operator, b_prepared, 10)

initial_approximation = (grid[0]-0.5)**2+(grid[1]-0.5)**2

evo_operators, meta_params = evolution(source_matrix=initial_approximation.cpu().detach().numpy(), crossover=partial(geo_crossover_fixed_box, box_size=10), fitness=fitness_function)

sln=np.genfromtxt('wave_sln_10.csv',delimiter=',')

mae = []
time_taken = []
for run_id in range(1):
    print(f'run_id: {run_id}')
    evo_strategy = BasicEvoStrategy(evo_operators=evo_operators, meta_params=meta_params,
                                        history=0, source_matrix=initial_approximation.cpu().detach().numpy(), real_solution=sln)
    start = time.time()
    evo_strategy.run()
    end = time.time()

    mae.append(evo_strategy.mae)
    time_taken.append(end-start)
    print(f'mae = {evo_strategy.mae}')
    print(f'Time taken = {end-start}')

