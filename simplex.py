import numpy as np


def print_problem(C, A, b, minimize=False):
    """
    Prints the optimization problem in standard form.
    """
    if minimize:
        print("Minimize: z = ", ' + '.join([f"{C[i]}*x{i+1}" for i in range(len(C))]))
    else:
        print("Maximize: z = ", ' + '.join([f"{C[i]}*x{i+1}" for i in range(len(C))]))
    
    print("Subject to:")
    for i in range(len(A)):
        print(' + '.join([f"{A[i][j]}*x{j+1}" for j in range(len(C))]), "<=", b[i])
    print()


def initialize_tableau(C, A, b):
    """
    Initializes the tableau by adding slack variables to the constraints
    and adding the objective function to the last row.
    """
    num_vars = len(C)
    num_constraints = len(A)
    
    # Add slack variables (identity matrix) to A
    tableau = np.hstack([A, np.eye(num_constraints), np.array(b).reshape(-1, 1)])
    
    # Add the objective function row (negate it for maximization)
    obj_row = np.hstack([-np.array(C), np.zeros(num_constraints + 1)])
    tableau = np.vstack([tableau, obj_row])
    
    return tableau


def pivot(tableau, pivot_row, pivot_col):
    """
    Performs the pivot operation on the tableau.
    """
    # Normalize the pivot row
    tableau[pivot_row] /= tableau[pivot_row][pivot_col]
    
    # Perform row operations to make all other entries in the pivot column 0
    for i in range(len(tableau)):
        if i != pivot_row:
            row_factor = tableau[i][pivot_col]
            tableau[i] -= row_factor * tableau[pivot_row]


def simplex_iterate(tableau, eps=1e-6):
    """
    Iterates through the Simplex method to optimize the objective function.
    """
    num_vars = tableau.shape[1] - 1  # Total number of columns (including slack + rhs)
    
    while any(tableau[-1, :-1] < -eps):
        # Step 1: Select the entering variable (most negative coefficient in the objective row)
        pivot_col = np.argmin(tableau[-1, :-1])
        
        # Step 2: Select the leaving variable (smallest positive ratio of RHS / pivot column entry)
        ratios = []
        for i in range(len(tableau) - 1):  # Ignore the objective row
            if tableau[i, pivot_col] > eps:  # Pivot column values must be positive
                ratios.append((tableau[i, -1] / tableau[i, pivot_col], i))
        
        if not ratios:
            # If no valid leaving variable, the problem is unbounded
            return "unbounded"
        
        # Step 3: Perform the pivot operation
        pivot_row = min(ratios)[1]
        pivot(tableau, pivot_row, pivot_col)
    
    return tableau


def extract_solution(tableau, num_vars):
    """
    Extracts the optimal solution from the final tableau.
    """
    solution = np.zeros(num_vars)
    for i in range(len(tableau) - 1):
        for j in range(num_vars):
            if tableau[i, j] == 1 and np.sum(tableau[:, j]) == 1:
                solution[j] = tableau[i, -1]
    
    z = tableau[-1, -1]  # Optimal value of the objective function
    return solution, z


def SimplexMethod(C, A, b, eps=1e-6, minimize=False):
    """
    The main Simplex method solver, supporting both maximization and minimization.
    """
    if minimize:
        C = [-c for c in C]  # Negate the objective function for minimization
    
    # Step 1: Print the problem
    print_problem(C if not minimize else [-c for c in C], A, b, minimize)
    
    # Step 2: Initialize the tableau
    tableau = initialize_tableau(C, A, b)
    print("Initial Tableau:\n", tableau, "\n")
    
    # Step 3: Perform the Simplex iterations
    result = simplex_iterate(tableau, eps)
    
    if (not isinstance(result, np.ndarray)) and (result == "unbounded"):
        print("The method is not applicable!")
    else:
        # Step 4: Extract the solution and objective value
        solution, z = extract_solution(tableau, len(C))
        if minimize:
            z = -z  # Negate the objective value for minimization
        print(f"Optimal solution: {solution}")
        print(f"Optimal value of the objective function: {z}")


print("------------------------------1---------------------------------")
C = [2, 3, 4]
A = [[1, 2, 3], [2, 2, 5], [4, 1, 2]]
b = [30, 24, 36]
SimplexMethod(C, A, b)  # Correct (`https://www.pmcalculators.com/simplex-method-calculator/?problem=('cr![-17938995847172.Aco!-9374.~s!-*~*~*Ar!-30724736.~o!'Maximizar')*≤'-['.']7'~8.,-927A]~A987.-*_`)
print("------------------------------2---------------------------------")
C = [-3, 1, 8]
A = [[3, 1, 4], [1, 2, 3], [5, 5, 4], [2, 4, 9]]
b = [18, 20, 7, 11]
SimplexMethod(C, A, b, minimize=True)  # Correct (`https://www.pmcalculators.com/simplex-method-calculator/?problem=('cr![63A1A4.1A2A3.5A5A4.2A4A9BCco!6-3A1A8B~s!6*~*~*~*Cr!618A20A7A11B~o!'Minimizar')*≤'.B,66['A'~B']C]~CBA6.*_`)
print("------------------------------3---------------------------------")
C = [1, 1]
A = [[1, -1], [-1, -1]]
b = [1, -2]
SimplexMethod(C, A, b)  # Correct (unbounded, `https://www.pmcalculators.com/simplex-method-calculator/?problem=('cr0*-1.,*1.]~co!*1.~s0'≤'~≥.~r!*2.~o!'Maximizar')*['1'~.']0![0.*_`)
print("------------------------------4---------------------------------")
C = [1, 1]
A = [[1, -1], [-1, -1], [2, 1]]
b = [1, -2, 3]
SimplexMethod(C, A, b)  # Correct (`https://www.pmcalculators.com/simplex-method-calculator/?problem=('cr![*-5*50241.]~co!*1.~s!0≤4≥4≤.~r!*243.~o!'Maximizar')*014.']0['4'~51.,540.*_`)
print("------------------------------5---------------------------------")
C = [-5, 4, 3, 7, -2]
A = [[3, 2, 1, 0, 4], [0, 5, 3, 6, 1], [2, 0, 4, 1, 5], [7, 3, 6, 2, 2]]
b = [25, 30, 20, 40]
SimplexMethod(C, A, b, minimize=True)  # Correct (`https://www.pmcalculators.com/simplex-method-calculator/?problem=('cr![93.2.1.B48B5C6.182.B4.1.587C6.2.2ADco!9-5.4C7.-2A~s!9*~*~*~*Dr!925.3B2B40A~o!'Minimizar')*≤'.'~8A,99['A']B0.C.3.D]~DCBA98.*_`)
