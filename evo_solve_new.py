from solver.solver_matrix import op_dict_to_list, operator_unify, bnd_prepare_matrix, matrix_loss
import torch
import numpy as np
import time
from MatrixEvolution.evo_operators import (
    k_point_crossover,
    increasing_dynamic_geo_crossover,
    fitness_frob_norm_only_u,
    fitness_combined_norm_only_u
)
from MatrixEvolution.evo_storage import EvoStorage
from MatrixEvolution.viz import joint_convergence_boxplots
from MatrixEvolution.init_randomly import (evolution_only_u_component, evo_random)
from functools import partial
from MatrixEvolution.evo_algorithm import (EvoHistory, BasicEvoStrategy)

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

