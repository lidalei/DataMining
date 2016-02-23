import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import dataloader_1b, random

# number of clusters
K = 4
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
                        [188, 189, 34], [23, 190, 207] ])


# return the index of the nearest point of p
def find_nearest_point(points, p):
    # initialize
    minimal_distance = euclidean(p, points[0])
    minimal_distance_point_index = 0
    
    for i in range(len(points)):
        distance = euclidean(p, points[i])
        if distance < minimal_distance:
            minimal_distance = distance
            minimal_distance_point_index = i
    
    return minimal_distance_point_index, minimal_distance


# compute k-means cost function
def k_medians_cost_function(points, k_centers, points_labels):
    cost_function = 0.0
    for i in range(len(points)):
        cost_function += euclidean(points[i], k_centers[points_labels[i]])
        
    return cost_function


def k_medians(points, k, initialization_method):
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
        
        # k_centers_indices = set([c0_index])
        
        distribution = np.zeros(len(points), dtype = np.float64)
        
        for r in range(1, len(k_centers)):
            for i in range(len(points)):
                nearest_center_index, nearest_distance = find_nearest_point(k_centers[0: r], points[i])
                distribution[i] = nearest_distance ** 2
            
            # normalization distribution
            sum_distances = sum(distribution)
            for i in range(len(distribution)):
                distribution[i] /= sum_distances
            
            # accumulate distribution
            accumulate_distribution = np.zeros(len(distribution), dtype = np.float64)
            accumulate_distribution[0] = distribution[0]
            for i in range(1, len(distribution)):
                accumulate_distribution[i] += distribution[i] + accumulate_distribution[i - 1]
            
            random_number = random.random()
            for i in range(len(accumulate_distribution)):
                if random_number <= accumulate_distribution[i] and accumulate_distribution[i] != 0:
                    k_centers[r] = points[i]
                    break
    
    elif initialization_method == GONZALES_ALGORITHM:
        print "GONZALES_ALGORITHM"
        
        c0_index = random.randint(0, len(points) - 1)
        k_centers[0] = points[c0_index]
        
        for t in range(1, len(k_centers)):
            
            t_th_center_index, cost_function = find_nearest_point(k_centers[0: t], points[0])
            
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
    k_medians_cost_function_values = []
    while True:
        # assignment
        for i in range(len(points)):
            nearest_center_index, nearest_distance = find_nearest_point(k_centers, points[i])
            points_labels[i] = nearest_center_index
        
        # compute k-means cost functions
        k_medians_cost_function_values.append(k_medians_cost_function(points, k_centers, points_labels))
        
        # update
        new_k_centers = np.zeros((len(k_centers), len(points[0])), dtype = np.float64)
        k_clusters = [[] for i in range(len(new_k_centers))]
        for j in range(len(points_labels)):
            k_clusters[points_labels[j]].append(points[j])
        
        # compute k-medians instead of k-means of each cluster
        # k-means
        # for i in range(len(new_k_centers)):
        #    new_k_centers[i] = np.mean(np.array(k_clusters[i]), axis = 0)
        # k-meidnas
        for i in range(len(new_k_centers)):
           new_k_centers[i] = np.median(np.array(k_clusters[i]), axis = 0)
              
        if np.linalg.norm(np.linalg.norm(new_k_centers - k_centers, axis = 1)) <= 10.0 ** (-10):
            break
        else:
            k_centers = new_k_centers
    
    return k_centers, points_labels, k_medians_cost_function_values

if __name__ == "__main__":
    print "k:", K
    
    points = dataloader_1b.load_data_1b(DATA_SET_FILE)
    k_centers, points_labels, k_medians_cost_function_values = k_medians(points, K, K_MEANS_PLUS_PLUS)
    
    print "Cost function:", k_medians_cost_function_values
    
    points_x = [p[0] for p in points]
    points_y = [p[1] for p in points]
    
    k_centers_x = [c[0] for c in k_centers]
    k_centers_y = [c[1] for c in k_centers]
    
    # plt.plot(points_x, points_y, ".", k_centers_x, k_centers_y, "r^")
    plt.scatter(points_x, points_y, c = [CATEGORY10[label] / 255.0 for label in points_labels], alpha = 0.8)
    plt.ylim([min(points_x), max(points_y) + 5])
    plt.show()