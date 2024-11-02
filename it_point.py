import numpy as np


def initialize_tableau(C, A, b):
    tableau = np.hstack([A, np.array(b).reshape(-1, 1)])
    obj_row = np.hstack([-np.array(C), 0])
    tableau = np.vstack([tableau, obj_row])
    return tableau

def pivot(tableau, pivot_row, pivot_col):
    tableau[pivot_row] /= tableau[pivot_row][pivot_col]
    for i in range(len(tableau)):
        if i != pivot_row:
            tableau[i] -= tableau[i][pivot_col] * tableau[pivot_row]

def simplex_iterate(tableau, eps=1e-6):
    while any(tableau[-1, :-1] < -eps):
        pivot_col = np.argmin(tableau[-1, :-1])
        ratios = [(tableau[i, -1] / tableau[i, pivot_col], i) for i in range(len(tableau) - 1) if tableau[i, pivot_col] > eps]
        if not ratios: return "unbounded"
        pivot_row = min(ratios)[1]
        pivot(tableau, pivot_row, pivot_col)
    return tableau

def extract_solution(tableau, num_vars):
    solution = np.zeros(num_vars)
    for i in range(len(tableau) - 1):
        for j in range(num_vars):
            if tableau[i, j] == 1 and np.sum(tableau[:, j]) == 1:
                solution[j] = tableau[i, -1]
    z = tableau[-1, -1]
    return solution, z

def SimplexMethod(C, A, b, eps=1e-6, minimize=False):
    if minimize: C = [-c for c in C]
    tableau = initialize_tableau(C, A, b)
    result = simplex_iterate(tableau, eps)
    if isinstance(result, str): return "The problem does not have a solution!"
    solution, z = extract_solution(tableau, len(C))
    if minimize: z = -z
    return solution, z


def interior_point(C, A, b, x, epsilon, alpha=0.5, minimize=False, max_iterations=100):
    # Step 0: Convert inputs to numpy arrays and validate input parameters
    try:
        C = np.array(C, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        x = np.array(x, dtype=float)

        m, n = A.shape
    except (TypeError, ValueError, AttributeError):
        return "The method is not applicable!"

    if b.shape != (m,) or C.shape != (n,) or x.shape != (n,):
        return "The method is not applicable!"

    if np.any(x < 0)  or not np.allclose(A @ x, b, atol=epsilon):
        return "The method is not applicable!"

    if minimize:
        C = -C

    i = 1
    while i <= max_iterations:
        v = x.copy()

        # Step 2: Create diagonal matrix D and perform transformations
        try:
            D = np.diag(x)
        except ValueError:
            return "The method is not applicable!"
        AA = A @ D
        cc = D @ C

        # Step 3: Create the projection matrix P
        try:
            AAT_inv = np.linalg.inv(AA @ AA.T)
            P = np.eye(n) - AA.T @ AAT_inv @ AA
        except np.linalg.LinAlgError:
            return "The method is not applicable!"

        # Step 4: Calculate cp, centering step
        cp = P @ cc
        nu = np.abs(np.min(cp))

        if np.isclose(nu, 0, atol=1e-7):
            return "The problem does not have a solution!"

        # Step 5: Update y and x using centering step
        y = np.ones(n) + (alpha / nu) * cp
        yy = D @ y
        x = yy

        # Step 6: Check for convergence
        if np.linalg.norm(yy - v, ord=2) < epsilon:
            z = C @ yy
            if minimize:
                z = -z
            return x, z

        i += 1

    return "The problem does not have a solution!"


# TODO: Add initial feasible solution to all tests
TESTS = [
    (
        [2, 3, 4],  # C
        [[1, 2, 3], [2, 2, 5], [4, 1, 2]],  # A
        [30, 24, 36],  # b
        False,  # minimize
        [1, 1, 1, 24, 15, 29]  # initial feasible solution (x + slacks)
    ),
    (
        [-3, 1, 8],
        [[3, 1, 4], [1, 2, 3], [5, 5, 4], [2, 4, 9]],
        [18, 20, 7, 11],
        True,
        [0.5, 0.5, 0.5, 14, 17, 0, 3.5]
    ),
    (
        [1, 5],
        [[1, 1], [-1, 3]],
        [10, 5],
        False,
        [1, 1, 8, 3]
    ),
    (
        [1, 1],
        [[1, -1], [-1, -1], [2, 1]],
        [1, -2, 3],
        False,
        []
    ),
    (
        [1, 1],
        [[1, -1], [-1, -1]],
        [1, -2],
        False,
        [3, 3, 1, 4]
    )
]


def convert_to_equality(C, A, b):
    C = np.array(C, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    num_constraints, num_vars = A.shape

    I = np.eye(num_constraints)
    A_eq = np.hstack((A, I))
    C_slack = np.zeros(num_constraints)
    C_eq = np.concatenate((C, C_slack))
    return A_eq, b, C_eq


def clean_solution(x, threshold=1e-4, decimals=3):
    x = np.where(np.abs(x) < threshold, 0, x)
    x = np.round(x, decimals)
    return x


def extract_answer(result: tuple | str) -> str:
    if isinstance(result, str):
        return result
    elif isinstance(result, tuple):
        solution, z_value = result
        solution = clean_solution(solution)
        z_value = np.round(z_value, 3)
        return f"X={solution}   z={z_value}"
    else:
        return "Unexpected result obtained"


def run_tests(epsilon=1e-4):
    for i in range(len(TESTS)):
        C, A, b, minimize, x_init = TESTS[i]
        A_eq, b_eq, C_eq = convert_to_equality(C, A, b)

        simplex_result = SimplexMethod(C_eq, A_eq, b_eq, eps=epsilon, minimize=minimize)

        if x_init:
            ip_result_alpha_05 = interior_point(C_eq, A_eq, b_eq, x_init, epsilon, alpha=0.5, minimize=minimize)
            ip_result_alpha_09 = interior_point(C_eq, A_eq, b_eq, x_init, epsilon, alpha=0.9, minimize=minimize)
        else:
            ip_result_alpha_05 = "PROVIDE INITIAL SOLUTION"
            ip_result_alpha_09 = "PROVIDE INITIAL SOLUTION"

        print(f"Results for Test {i+1}:")
        print(f"  Simplex Result:                     {extract_answer(simplex_result)}")
        print(f"  Interior-Point (alpha=0.5) Result:  {extract_answer(ip_result_alpha_05)}")
        print(f"  Interior-Point (alpha=0.9) Result:  {extract_answer(ip_result_alpha_09)}")
        print()


def main():
    run_tests()

if __name__ == "__main__":
    main()
