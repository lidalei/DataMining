import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import dataloader_1b, random

# initialization methods
FIRST_K_POINTS = 1
UNIFORMLY_K_POINTS = 2
K_MEANS_PLUS_PLUS = 3
GONZALES_ALGORITHM = 4
# data set file name
DATA_SET_FILE = "data1b/C2.txt"
# clusters colors
CATEGORY10 = np.array([ [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
                        [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
                        [188, 189, 34], [23, 190, 207] ]) / 255.0


# return the index of the nearest point of p
def find_nearest_point(points, p):
    # initialize
    minimal_distance = euclidean(p, points[0])
    minimal_distance_point_index = 0
    
    for i in range(1, len(points)):
        distance = euclidean(p, points[i])
        if distance < minimal_distance:
            minimal_distance = distance
            minimal_distance_point_index = i
    
    return minimal_distance_point_index, minimal_distance


# compute k-means cost function
def k_means_cost_function(points, k_centers, points_labels):
    cost_function = 0.0
    for i in range(len(points)):
        cost_function += euclidean(points[i], k_centers[points_labels[i]]) ** 2
        
    return cost_function


def k_means(points, k, initialization_method):
    if k <= 0 or len(points) <= k:
        return False
    # initialize k centers with zeroes
    k_centers = np.zeros((k, len(points[0])), dtype = np.float64)
    
    # initialization
    if initialization_method == FIRST_K_POINTS:
        print "FIRST_K_POINTS"
        
        k_centers = points[0:k]
    
    elif initialization_method == UNIFORMLY_K_POINTS:
        print "UNIFORMLY_K_POINTS"
        
        random_array = np.zeros(len(points), dtype = np.int)
        for i in range(random_array.size - 1):
            random_array[i + 1] =  random_array[i] + 1
        # permute to generate random array
        for i in range(random_array.size):
            j = random.randint(0, random_array.size - 1)
            e = random_array[i]
            random_array[i] = random_array[j]
            random_array[j] = e
            
        for i in range(len(k_centers)):
            k_centers[i] = points[random_array[i]]
                
    elif initialization_method == K_MEANS_PLUS_PLUS:
        print "K_MEANS_PLUS_PLUS"
        
        c0_index = random.randint(0, len(points) - 1)
        k_centers[0] = points[c0_index]
        
        distribution = np.zeros(len(points), dtype = np.float64)
        
        for r in range(1, len(k_centers)):
            for i in range(len(points)):
                nearest_center_index, nearest_distance = find_nearest_point(k_centers[0: r], points[i])
                distribution[i] = nearest_distance ** 2
            
            # normalization distribution
            sum_distances = np.sum(distribution)
            distribution /= sum_distances
            
            # accumulate distribution
            accumulate_distribution = np.zeros(len(distribution), dtype = np.float64)
            accumulate_distribution[0] = distribution[0]
            for i in range(1, len(distribution)):
                accumulate_distribution[i] = distribution[i] + accumulate_distribution[i - 1]
            
            random_number = random.random()
            for i in range(len(accumulate_distribution)):
                if random_number <= accumulate_distribution[i] and distribution[i] != 0:
                    k_centers[r] = points[i]
                    break
    
    elif initialization_method == GONZALES_ALGORITHM:
        print "GONZALES_ALGORITHM"
        
        # c0_index = random.randint(0, len(points) - 1)
        # k_centers[0] = points[c0_index]
        k_centers[0] = points[0]
        
        for t in range(1, len(k_centers)):
            
            nearest_center_index, cost_function = find_nearest_point(k_centers[0: t], points[0])
            t_th_center_index = 0
            for i in range(1, len(points)):
                nearest_center_index, nearest_distance = find_nearest_point(k_centers[0: t], points[i])
                         
                if nearest_distance > cost_function:
                    t_th_center_index = i
                    cost_function = nearest_distance
             
            k_centers[t] = points[t_th_center_index]
        
    else:
        return False
          
    # clustering
    # initialize k clusters, i.e., label array
    points_labels = np.zeros(len(points), dtype = np.int)
    k_means_cost_function_values = []
    while True:
        # assignment
        for i in range(len(points)):
            nearest_center_index, nearest_distance = find_nearest_point(k_centers, points[i])
            points_labels[i] = nearest_center_index
        
        # compute k-means cost functions
        k_means_cost_function_values.append(k_means_cost_function(points, k_centers, points_labels))
        
        # update
        new_k_centers = np.zeros((len(k_centers), len(points[0])), dtype = np.float64)
        sums = np.zeros((len(k_centers), len(points[0])), dtype = np.float64)
        counts = np.zeros(len(k_centers), dtype = np.int)
        for i in range(len(points_labels)):
            sums[points_labels[i]] = np.add(sums[points_labels[i]], points[i])
            counts[points_labels[i]] += 1
        for i in range(len(new_k_centers)):
            for j in range(len(points[0])):
                new_k_centers[i][j] = sums[i][j] / counts[i]
            
        if np.linalg.norm(np.linalg.norm(new_k_centers - k_centers, axis = 1)) <= 10.0 ** (-10):
            k_centers = new_k_centers
            k_means_cost_function_values.append(k_means_cost_function(points, k_centers, points_labels))
            break
        else:
            k_centers = new_k_centers
    
    return k_centers, points_labels, k_means_cost_function_values

if __name__ == "__main__":
    points = dataloader_1b.load_data_1b(DATA_SET_FILE)
    points_x = [p[0] for p in points]
    points_y = [p[1] for p in points]
        
    # number of clusters
    for k in [3, 4, 5]:
        print "k:", k
        
        costs_different_initializations = {}
        
        ## initialize with first k points in the data-set
        k_centers, points_labels, k_means_cost_function_values = k_means(points, k, FIRST_K_POINTS)
        costs_different_initializations["FIRST_K_POINTS"] = k_means_cost_function_values
        
        print "Cost function", k_means_cost_function_values
           
        k_centers_x = [c[0] for c in k_centers]
        k_centers_y = [c[1] for c in k_centers]
        
        fig1, (axis_clusters, axis_cost) = plt.subplots(1, 2)
        axis_clusters.scatter(points_x, points_y, c = [CATEGORY10[label] for label in points_labels], alpha = 0.8, label = "clusters")
        axis_clusters.scatter(k_centers_x, k_centers_y, marker = "+", label = "centers")
        axis_clusters.set_ylim([min(points_x), max(points_y) + 5])
        axis_clusters.set_title("Clusters with first " + str(k) + " points initialization")
        axis_clusters.legend(loc = "upper right", fontsize = "medium")
        
        axis_cost.plot(k_means_cost_function_values)
        axis_cost.set_title("Cost function with first " + str(k) + " points initialization")
        axis_cost.set_xlabel("Number of iterations")
        axis_cost.set_ylabel("Cost function")
        
        ## initialize with k points uniformly picked at random
        costs_different_initializations["UNIFORMLY_K_POINTS"] = []
        k_centers, points_labels, k_means_cost_function_values = None, None, None
        for i in range(5):
            print "Run", i + 1
            
            k_centers, points_labels, k_means_cost_function_values = k_means(points, k, UNIFORMLY_K_POINTS)
            costs_different_initializations["UNIFORMLY_K_POINTS"].append(k_means_cost_function_values)
        
            print "Cost function", k_means_cost_function_values
        
        # compute average and standard deviation of final costs of different runs
        final_costs = np.array([costs_i[-1] for costs_i in costs_different_initializations["UNIFORMLY_K_POINTS"]])
        print "Average:", np.average(final_costs), ", Standard deviation:", np.std(final_costs) 
            
        k_centers_x = [c[0] for c in k_centers]
        k_centers_y = [c[1] for c in k_centers]
        
        fig2, (axis_clusters, axis_cost) = plt.subplots(1, 2)
        axis_clusters.scatter(points_x, points_y, c = [CATEGORY10[label] for label in points_labels], alpha = 0.8, label = "clusters")
        axis_clusters.scatter(k_centers_x, k_centers_y, marker = "+", label = "centers")
        axis_clusters.set_ylim([min(points_x), max(points_y) + 5])
        axis_clusters.set_title("Clusters with uniformly picked " + str(k) + " points initialization")
        axis_clusters.legend(loc = "upper right", fontsize = "medium")
        
        for i in range(5):
            axis_cost.plot(costs_different_initializations["UNIFORMLY_K_POINTS"][i], label = "UNIFORMLY_K_POINTS" + "_" + str(i + 1))
        axis_cost.legend(loc = "upper right", fontsize = "medium")
        axis_cost.set_title("Cost function with uniformly picked " + str(k) + " points initialization")
        axis_cost.set_xlabel("Number of iterations")
        axis_cost.set_ylabel("Cost function")
            
        ## initialize with k-means++
        costs_different_initializations["K_MEANS_PLUS_PLUS"] = []
        k_centers, points_labels, k_means_cost_function_values = None, None, None
        for i in range(5):
            print "Run", i + 1
            k_centers, points_labels, k_means_cost_function_values = k_means(points, k, K_MEANS_PLUS_PLUS)
            costs_different_initializations["K_MEANS_PLUS_PLUS"].append(k_means_cost_function_values)
            
            print "Cost function", k_means_cost_function_values
        
        # compute average and standard deviation of final costs of different runs
        final_costs = np.array([costs_i[-1] for costs_i in costs_different_initializations["K_MEANS_PLUS_PLUS"]])
        print "Average:", np.average(final_costs), ", Standard deviation:", np.std(final_costs)
           
        k_centers_x = [c[0] for c in k_centers]
        k_centers_y = [c[1] for c in k_centers]
        
        fig3, (axis_clusters, axis_cost) = plt.subplots(1, 2)
        axis_clusters.scatter(points_x, points_y, c = [CATEGORY10[label] for label in points_labels], alpha = 0.8, label = "clusters")
        axis_clusters.scatter(k_centers_x, k_centers_y, marker = "+", label = "centers")
        axis_clusters.set_ylim([min(points_x), max(points_y) + 5])
        axis_clusters.set_title("Clusters with k-means++, k= " + str(k))
        axis_clusters.legend(loc = "upper right", fontsize = "medium")
        for i in range(5):
            axis_cost.plot(costs_different_initializations["K_MEANS_PLUS_PLUS"][i], label = "K_MEANS_PLUS_PLUS" + "_" + str(i + 1))
        axis_cost.legend(loc = "upper right", fontsize = "medium")
        axis_cost.set_title("Cost function with k-means++, k= " + str(k))
        axis_cost.set_xlabel("Number of iterations")
        axis_cost.set_ylabel("Cost function")
        
        ## initialize with GONZALES' algorithm
        k_centers, points_labels, k_means_cost_function_values = k_means(points, k, GONZALES_ALGORITHM)
        costs_different_initializations["GONZALES_ALGORITHM"] = k_means_cost_function_values
        
        print "Cost function", k_means_cost_function_values
          
        k_centers_x = [c[0] for c in k_centers]
        k_centers_y = [c[1] for c in k_centers]
        
        fig4, (axis_clusters, axis_cost) = plt.subplots(1, 2)
        axis_clusters.scatter(points_x, points_y, c = [CATEGORY10[label] for label in points_labels], alpha = 0.8, label = "clusters")
        axis_clusters.scatter(k_centers_x, k_centers_y, marker = "+", label = "centers")
        axis_clusters.set_ylim([min(points_x), max(points_y) + 5])
        axis_clusters.set_title("Clusters with GONZALES' algorithm initialization, k= " + str(k))
        axis_clusters.legend(loc = "upper right", fontsize = "medium")
        
        axis_cost.plot(k_means_cost_function_values)
        axis_cost.set_title("Cost function with GONZALES' algorithm initialization, k= " + str(k))
        axis_cost.set_xlabel("Number of iterations")
        axis_cost.set_ylabel("Cost function")
        
        ## plot the cost function comparison
        fig5, axis_costs = plt.subplots(1, 1)
        for key in costs_different_initializations:
            costs = costs_different_initializations[key]
            if all(isinstance(costs_i, list) for costs_i in costs):
                # for i in range(len(costs)):
                    # axis_costs.plot(range(1, len(costs[i]) + 1), costs[i], label = key + "_" + str(i + 1))
                axis_costs.plot(range(1, len(costs[0]) + 1), costs[0], label = key)
            else:
                axis_costs.plot(range(1, len(costs) + 1), costs, label = key)
            
        axis_costs.legend(loc = "upper right", fontsize = "medium")
        axis_costs.set_title("Cost function with different initializations, k= " + str(k))
        axis_costs.set_xlabel("Number of iterations")
        axis_costs.set_ylabel("Cost function")
        
        plt.show()