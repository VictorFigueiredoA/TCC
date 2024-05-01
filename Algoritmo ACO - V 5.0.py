import pandas as pd
import numpy as np
from random import randrange

# Read the Excel file into a pandas DataFrame
# d = pd.read_excel('C:/Users/nickb/Downloads/Tab24_OD2017_2.xlsx', header=0, index_col=0)
# d = pd.read_excel('C:/Users/nickb/Downloads/Tab24_OD2017.xlsx', header=0, index_col=0)
d = pd.read_excel('AAA4_2_T.xlsx', header=0, index_col=0)
d

pd.option_context('display.height', 500, 'display.max_rows', 500)
# d = d.replace(0, 99999)
# d.values[[np.arange(d.shape[0])]*2] = 0
# d = d.round(0) # Arredodamento
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
            
    def _empty_next_nodes(self, current_node):
        empty_list=[]
        empty_list = list(np.where(distance_matrix[current_node] == 0)[0])
#         print(empty_list)
        return empty_list
            

    def _select_next_node(self, current_node, visited_nodes, empty_list):
        pheromone_values = self.pheromone_matrix[current_node].copy()
#         print(pheromone_values)
        pheromone_values[empty_list] = 0
        pheromone_values[list(visited_nodes)] = 0
#         print(pheromone_values, self.alpha)
#         print(self.distance_matrix[current_node], self.beta)
        probabilities = pheromone_values ** self.alpha / (self.distance_matrix[current_node] ** self.beta)
#         print(probabilities)
        # Handle NaN or Inf values by setting them to zero
        probabilities[np.isnan(probabilities) | np.isinf(probabilities)] = 0

        # Normalize probabilities to avoid division by zero
        if probabilities.sum() > 0:
            probabilities /= probabilities.sum()
            next_node = np.random.choice(range(self.num_nodes), p=probabilities)
        if probabilities.sum() == 0:
            next_node = -1
        
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
                    empty_list = self._empty_next_nodes(current_node)
#                     print(current_node)
#                     print(empty_list)
                    next_node = self._select_next_node(current_node, visited_nodes, empty_list) #4
                    if next_node == -1:
                        total_distance = np.inf
                        break
                    else:
                        visited_nodes.add(next_node)
                        path.append(next_node) # 3-0-1-4-2
                        total_distance += self.distance_matrix[current_node][next_node] #12+10+35=57+2=59
                        current_node = next_node 
                if next_node != -1:
                    ants.append({'path': path, 'distance': total_distance})
                if total_distance < best_ant['distance']:
                    if next_node != -1:
                        best_ant = {'path': path, 'distance': total_distance}
            self._update_pheromones(ants, best_ant)
        return best_ant['path'], best_ant['distance']

distance_matrix = np.array(d)
    
# Initialize ACO parameters
num_ants = 5
#analise
num_iterations = 100 
evaporation_rate = 0.01
    
    
# Create ACO instance
aco = AntColonyOptimization(distance_matrix, num_ants, num_iterations, evaporation_rate)


for i in range(0,9):
    
    # Find optimal path
    start_node = randrange(0, 4)
    end_node = randrange(0, 4)
    while end_node == start_node:
        end_node = randrange(0, 4)

    optimal_path, optimal_distance = aco.find_path(start_node, end_node) # Ajuste indice -> número do caminho
    optimal_path_volta, optimal_distance_volta = aco.find_path(end_node, start_node) # Ajuste indice -> número do caminho
    print("Optimal path ida:", optimal_path)
    print("Optimal distance ida:", optimal_distance)
    print("Optimal path volta:", optimal_path_volta)
    print("Optimal distance volta:", optimal_distance_volta)
    print("Custo Total:", optimal_distance+optimal_distance_volta)
    print("____________________________________")
    
    
start_node_final = 0
end_node_final = 3
optimal_path, optimal_distance = aco.find_path(start_node_final, end_node_final)
print("---------------------------------------------")
print("Optimal path ida:", optimal_path)
print("Optimal distance ida:", optimal_distance)



