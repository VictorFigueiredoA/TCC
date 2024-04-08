import numpy as np
import pandas as pd

# Read the Excel file into a pandas DataFrame
d = pd.read_excel("AAA2.xlsx")

print(d.head())

iteration = 100
n_ants = 3
n_citys = 3

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
pheromone = 0.1 * np.ones((m, n))

# initializing the route of the ants with size route(n_ants, n_citys + 1)
# note adding 1 because we want to come back to the source city
route = np.ones((m, n + 1))

for ite in range(iteration):
    route[:, 0] = 1  # initial starting and ending position of every ant '1' i.e city '1'
    for i in range(m):
        temp_visibility = np.array(visibility)  # creating a copy of visibility
        for j in range(n - 1):
            combine_feature = np.zeros(n)  # initializing combine_feature array to zero
            cum_prob = np.zeros(n)  # initializing cumulative probability array to zeros
            cur_loc = int(route[i, j] - 1)  # current city of the ant
            temp_visibility[:, cur_loc] = 0  # making visibility of the current city as zero
            p_feature = np.power(pheromone[cur_loc, :], beta)  # calculating pheromone feature
            v_feature = np.power(temp_visibility[cur_loc, :], alpha)  # calculating visibility feature
            p_feature = p_feature[:, np.newaxis]  # adding axis to make a size[5,1]
            v_feature = v_feature[:, np.newaxis]  # adding axis to make a size[5,1]
            combine_feature = np.multiply(p_feature, v_feature)  # calculating the combine feature
            total = np.sum(combine_feature)  # sum of all the feature
            probs = combine_feature / total  # finding probability of element probs(i) = combine_feature(i)/total
            cum_prob = np.cumsum(probs)  # calculating cumulative sum
            r = np.random.random_sample()  # random no in [0,1)
            city = np.nonzero(cum_prob > r)[0][0] + 1  # finding the next city having probability higher than random(r)
            route[i, j + 1] = city  # adding city to route
        left = list(set(range(1, n + 1)) - set(route[i, :-2]))[0]  # finding the last untraversed city in route
        route[i, -2] = left  # adding untraversed city to route
    route_opt = np.array(route)  # initializing optimal route
    dist_cost = np.zeros((m, 1))  # initializing total_distance_of_tour with zero
    for i in range(m):
        s = 0
        for j in range(n - 1):
            s = s + d.values[int(route_opt[i, j]) - 1, int(route_opt[i, j + 1]) - 1]  # calculating total tour distance
        dist_cost[i] = s  # storing distance of tour for 'i'th ant at location 'i'
    dist_min_loc = np.argmin(dist_cost)  # finding location of minimum of dist_cost
    dist_min_cost = dist_cost[dist_min_loc]  # finding min of dist_cost
    best_route = route[dist_min_loc, :]  # initializing current traversed as best route
    pheromone = (1 - e) * pheromone  # evaporation of pheromone with (1-e)
    for i in range(m):
        for j in range(n - 1):
            pheromone[int(route_opt[i, j]) - 1, int(route_opt[i, j + 1]) - 1] += 1 / dist_cost[i]  # updating pheromone with delta_distance
            # delta_distance will be more with min_dist i.e adding more weight to that route pheromone

print('route of all the ants at the end :')
print(route_opt)
print()
print('best path :', best_route)
print('cost of the best path', int(dist_min_cost[0]) + d.values[int(best_route[-2]) - 1, 0])