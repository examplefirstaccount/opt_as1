# Introduction to Optimization - Assignments

This repository contains implementations of optimization algorithms for solving Linear Programming Problems (LPP) as part of the "Introduction to Optimization" course. Each assignment focuses on a different method for solving LPPs, using pure NumPy for numerical computations.

## Assignments

### Assignment 1: Simplex Method

**File**: `simplex.py`  
**Main Function**: `SimplexMethod(C, A, b, eps=1e-6, minimize=False)`

This assignment implements the Simplex Method to solve LPPs with inequality constraints. The function `SimplexMethod` maximizes the objective function $z = C^T x$ subject to $Ax \leq b$. By default, the function maximizes $z$, but minimization can be specified using the `minimize` parameter.

- **Input**:
    - `C`: Coefficient vector of the objective function.
    - `A`: Coefficient matrix for constraints.
    - `b`: Right-hand side vector for constraints.
    - `eps`: Tolerance level for the algorithm (default is $1 \times 10^{-6}$).
    - `minimize`: Boolean flag for minimization (default is `False` for maximization).

- **Output**:
    - Prints the optimal solution and the optimal objective function value.

### Assignment 2: Interior-Point Method

**File**: `it_point.py`  
**Main Functions**: `interior_point(C, A, b, x, epsilon, alpha=0.5, minimize=False, max_iterations=1000)` and `SimplexMethod(C, A, b, eps=1e-6, minimize=False)`

In this assignment, both the Interior-Point and Simplex methods are implemented to solve LPPs with equality constraints. The objective is to maximize $z = C^T x$ subject to $Ax = b$ and $x \geq 0$. Both functions return the optimal solution and the optimal objective function value.

- **Input for `interior_point`**:
    - `C`: Coefficient vector of the objective function.
    - `A`: Coefficient matrix for constraints.
    - `b`: Right-hand side vector for constraints.
    - `x`: Initial guess for the solution (feasible).
    - `epsilon`: Convergence tolerance level.
    - `alpha`: Step size parameter (default is 0.5).
    - `minimize`: Boolean flag for minimization (default is `False` for maximization).
    - `max_iterations`: Maximum number of iterations (default is 1000).

- **Output**:
    - Returns the optimal solution and the optimal objective function value.

## Requirements

- Python 3.x
- NumPy

## How to Use

1. Clone the repository.
2. Import the required function from the relevant file.
3. Pass in your problem data (objective coefficients, constraint matrix, etc.) to obtain the optimal solution.
