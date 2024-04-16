import numpy as np
import pandas as pd

# Read the Excel file into a pandas DataFrame
d = pd.read_excel("AAA4.xlsx")

# Substituir valores zero na matriz de distância por um valor máximo
d.replace(0, np.inf, inplace=True)

iteration = 10
n_ants = 10
n_citys = 5

# initialization part
m = n_ants
n = n_citys
e = 0.5         # evaporation rate
alpha = 1       # pheromone factor
beta = 2        # visibility factor

# calculating the visibility of the next city visibility(i,j) = 1/d(i,j)
visibility = 1 / d.values
visibility[visibility == np.inf] = 0
# initializing pheromone present at the paths to the cities
pheromone = 0.1 * np.ones((n, n))

# Get user input for origin and destination
start_point = 1
end_point = 4  # Assuming destination point is fixed, change if needed

# initializing the route of the ants with size route(n_ants, n_citys + 1)
# note adding 1 because we want to come back to the source city
route = np.ones((m, n + 1))

# Initialize variables to store the best route and its cost
best_global_route = None
best_global_cost = np.inf

# Main ACO loop
for ite in range(iteration):
    for i in range(m):
        cur_loc = start_point - 1  # current city of the ant
        route[i, 0] = start_point  # set starting point
        for j in range(n):
            # Calculate combined feature
            combine_feature = (pheromone[cur_loc, :] ** beta) * (visibility[cur_loc, :] ** alpha)
            combine_feature[cur_loc] = 0  # Mask out the current city
            # Choose next city based on probability
            probabilities = combine_feature / np.sum(combine_feature)
            next_city = np.random.choice(np.arange(n) + 1, p=probabilities)
            # Update route
            route[i, j + 1] = next_city
            cur_loc = next_city - 1
            if next_city == end_point:  # Check if reached the destination
                # Finish the route if the destination is reached
                route[i, j + 2:] = end_point
                break  # Stop the loop if reached the destination

    # Calculate the total cost of all routes
    total_costs = []
    for i in range(m):
        total_cost = sum(d.values[int(route[i, j]) - 1, int(route[i, j + 1]) - 1] for j in range(n) if route[i, j + 1] != end_point)
        total_costs.append(total_cost)

    # Update global best route if necessary
    min_cost_index = np.argmin(total_costs)
    if total_costs[min_cost_index] < best_global_cost:
        best_global_cost = total_costs[min_cost_index]
        best_global_route = route[min_cost_index].copy()

# Print the best route found
print('Melhor rota encontrada:')
print(best_global_route)
print('Custo total:', int(best_global_cost))