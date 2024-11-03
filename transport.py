import numpy as np
import pandas as pd


def is_balanced(supply, demand):
    if sum(supply) == sum(demand):
        return True
    return False


def print_input_table(cost_matrix, supply, demand):
    df = pd.DataFrame(
        cost_matrix,
        columns=[f"D{j+1}" for j in range(len(demand))],
        index=[f"S{i+1}" for i in range(len(supply))]
    )

    print("\n==== Cost Matrix (C) ====")
    print(df)
    print("\nSupply Vector (S):", supply)
    print("Demand Vector (D):", demand)
    print("=" * 30)


# North-West Corner Method
def north_west_corner_method(supply, demand):
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))
    i, j = 0, 0

    while i < m and j < n:
        if supply[i] < demand[j]:
            allocation[i][j] = supply[i]
            demand[j] -= supply[i]
            i += 1
        else:
            allocation[i][j] = demand[j]
            supply[i] -= demand[j]
            j += 1

    return allocation


# Vogel’s Approximation Method
def vogel_approximation_method(supply, demand, cost_matrix):
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))

    while sum(supply) > 0 and sum(demand) > 0:
        row_penalties = []
        col_penalties = []

        # Calculate penalties for rows
        for i in range(m):
            if supply[i] > 0:
                available_costs = [cost_matrix[i][j] for j in range(n) if demand[j] > 0]
                if len(available_costs) > 1:
                    sorted_costs = sorted(available_costs)
                    row_penalties.append(sorted_costs[1] - sorted_costs[0])
                else:
                    row_penalties.append(0)
            else:
                row_penalties.append(-1)

        # Calculate penalties for columns
        for j in range(n):
            if demand[j] > 0:
                available_costs = [cost_matrix[i][j] for i in range(m) if supply[i] > 0]
                if len(available_costs) > 1:
                    sorted_costs = sorted(available_costs)
                    col_penalties.append(sorted_costs[1] - sorted_costs[0])
                else:
                    col_penalties.append(0)
            else:
                col_penalties.append(-1)

        # Find maximum penalty
        max_row_penalty = max(row_penalties)
        max_col_penalty = max(col_penalties)

        if max_row_penalty >= max_col_penalty:
            i = row_penalties.index(max_row_penalty)
            j = np.argmin([cost_matrix[i][k] if demand[k] > 0 else float('inf') for k in range(n)])
        else:
            j = col_penalties.index(max_col_penalty)
            i = np.argmin([cost_matrix[k][j] if supply[k] > 0 else float('inf') for k in range(m)])

        # Allocate
        allocation_amount = min(supply[i], demand[j])
        allocation[i][j] = allocation_amount
        supply[i] -= allocation_amount
        demand[j] -= allocation_amount

        # Mark exhausted supply or demand
        if supply[i] == 0:
            for k in range(n):
                cost_matrix[i][k] = float('inf')
        if demand[j] == 0:
            for k in range(m):
                cost_matrix[k][j] = float('inf')

    return allocation


# Russell’s Approximation Method
def russell_approximation_method(supply, demand, cost_matrix):
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))
    closed_list = set()

    while sum(supply) > 0 and sum(demand) > 0:
        # Compute heuristic values for each cell
        heuristics = float('inf'), -1, -1

        for i in range(m):
            for j in range(n):
                if (i, j) in closed_list:
                    continue

                # Compute h_ij = cost[i][j] - max(costs in row i) - max(costs in column j)
                max_row_cost = max(cost_matrix[i])
                max_col_cost = max([cost_matrix[k][j] for k in range(m)])
                h_ij = cost_matrix[i][j] - max_row_cost - max_col_cost
                if h_ij < heuristics[0]:
                    heuristics = h_ij, i, j

        # Find the cell with the most negative heuristic value
        min_value, i, j = heuristics
        if min_value == float('inf'):
            break

        # Allocate as much as possible to this cell
        allocation_amount = min(supply[i], demand[j])
        allocation[i][j] = allocation_amount
        supply[i] -= allocation_amount
        demand[j] -= allocation_amount

        closed_list.add((i, j))

        # Step 4: If supply or demand is zero, mark the entire row or column as closed
        if supply[i] == 0:
            for col in range(n):
                closed_list.add((i, col))
        if demand[j] == 0:
            for row in range(m):
                closed_list.add((row, j))

    return allocation


def transportation_problem(supply, demand, cost_matrix):
    if not is_balanced(supply, demand):
        print("The problem is not balanced!")
        return

    print_input_table(cost_matrix, supply, demand)

    cost_matrix = np.array(cost_matrix, dtype=float)

    nw_allocation = north_west_corner_method(supply.copy(), demand.copy())
    print("\nNorth-West Corner Method Allocation:")
    print(nw_allocation)

    vogel_allocation = vogel_approximation_method(supply.copy(), demand.copy(), np.copy(cost_matrix))
    print("\nVogel’s Approximation Method Allocation:")
    print(vogel_allocation)

    russell_allocation = russell_approximation_method(supply.copy(), demand.copy(), np.copy(cost_matrix))
    print("\nRussell’s Approximation Method Allocation:")
    print(russell_allocation)


TESTS = [
    (
        [20, 30, 50],  # Supply
        [30, 20, 40, 10],  # Demand
        [[8, 6, 10, 9], [9, 12, 13, 7], [14, 9, 16, 5]]  # Costs
    ),
    (
        [15, 25, 40],
        [20, 25, 20, 15],
        [[4, 8, 8, 6], [6, 10, 7, 5], [5, 7, 6, 8]]
    ),
    (
        [25, 35, 40],
        [20, 30, 30, 20],
        [[10, 4, 9, 7], [3, 8, 5, 6], [6, 9, 12, 10]]
    )
]


def run_tests():
    for i in range(len(TESTS)):
        supply, demand, cost_matrix = TESTS[i]
        print(f"******** Test Case {i+1} ********")
        transportation_problem(supply, demand, cost_matrix)
        print("\n\n")


if __name__ == "__main__":
    run_tests()
