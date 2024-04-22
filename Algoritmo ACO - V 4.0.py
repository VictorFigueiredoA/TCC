import pandas as pd
import numpy as np

# Read the Excel file into a pandas DataFrame
# d = pd.read_excel('C:/Users/nickb/Downloads/Tab24_OD2017_2.xlsx', header=0, index_col=0)
d = pd.read_excel('AAA4.xlsx', header=0, index_col=0)
# d = pd.read_excel('C:/Users/nickb/Downloads/AAA4.xlsx', header=0)

matriz = d.values

linhas, colunas = d.shape

if linhas == colunas:
    for i in range(linhas):
        matriz[i,i] = 0

d


class AntColonyOptimization:
    def __init__(self, distance_matrix, num_ants, num_iterations, evaporation_rate, alpha=1, beta=1):
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.num_nodes = len(distance_matrix)
        self.pheromone_matrix = np.ones((self.num_nodes, self.num_nodes))
        np.fill_diagonal(self.pheromone_matrix, 0)

    def _update_pheromones(self, ants, best_ant):
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        for ant in ants:
            path = ant['path']
            for i in range(len(path) - 1):
                current_node, next_node = path[i], path[i+1]
                self.pheromone_matrix[current_node][next_node] += 1 / ant['distance']
        best_path = best_ant['path']
        for i in range(len(best_path) - 1):
            current_node, next_node = best_path[i], best_path[i+1]
            self.pheromone_matrix[current_node][next_node] += 1 / best_ant['distance']

    def _select_next_node(self, current_node, visited_nodes):
        pheromone_values = self.pheromone_matrix[current_node].copy()
        pheromone_values[list(visited_nodes)] = 0
        probabilities = pheromone_values ** self.alpha / (self.distance_matrix[current_node] ** self.beta)

        # Handle NaN or Inf values by setting them to zero
        probabilities[np.isnan(probabilities) | np.isinf(probabilities)] = 0

        # Normalize probabilities to avoid division by zero
        if probabilities.sum() > 0:
            probabilities /= probabilities.sum()

        next_node = np.random.choice(range(self.num_nodes), p=probabilities)
        return next_node

    def find_path(self, start_node, end_node):
        best_ant = {'path': [], 'distance': float('inf')}
        for _ in range(self.num_iterations):
            ants = []
            for _ in range(self.num_ants):
                current_node = start_node
                visited_nodes = {current_node}
                path = [current_node]
                total_distance = 0
                while current_node != end_node:
                    next_node = self._select_next_node(current_node, visited_nodes)
                    visited_nodes.add(next_node)
                    path.append(next_node)
                    total_distance += self.distance_matrix[current_node][next_node]
                    current_node = next_node
                ants.append({'path': path, 'distance': total_distance})
                if total_distance < best_ant['distance']:
                    best_ant = {'path': path, 'distance': total_distance}
            self._update_pheromones(ants, best_ant)
        return best_ant['path'], best_ant['distance']

distance_matrix = np.array(d)
    
# Initialize ACO parameters
num_ants = 10
num_iterations = 100
evaporation_rate = 0.1   
    
    
# Create ACO instance
aco = AntColonyOptimization(distance_matrix, num_ants, num_iterations, evaporation_rate)

# Find optimal path
start_node = 4
end_node = 2
optimal_path, optimal_distance = aco.find_path(start_node, end_node) # Ajuste indice -> número do caminho
print("Optimal path:", optimal_path)
print("Optimal distance:", optimal_distance)