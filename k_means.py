import numpy as np
from scipy.spatial.distance import euclidean
import dataloader_1b, random

FIRST_K_POINTS = 1
UNIFORMLY_K_POINTS = 2
K_MEANS_PLUS_PLUS = 3
GONZALES_ALGORITHM = 4

c2 = dataloader_1b.load_data_1b("data1b/C2.txt")
# print c2

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

def k_means(points, k, initialization_method):
    if k <= 0 or len(points) <= k:
        return False
    # initialize k centers with zeroes
    k_centers = np.zeros((k, 2), dtype = np.float64)
    k_clusters = np.zeros(len(points), dtype = np.float64)
    
    # initialization
    if initialization_method == FIRST_K_POINTS:
        print "FIRST_K_POINTS"
        
        k_centers = points[0:k]
                
    elif initialization_method == UNIFORMLY_K_POINTS:
        print "UNIFORMLY_K_POINTS"
        
        random_array = np.zeros(len(points), dtype = np.int)
        for i in range(0, random_array.size - 1):
            random_array[i + 1] =  random_array[i] + 1
        for i in range(0, random_array.size - 1):
            j = random.randint(1, 10)
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
                nearest_point_index, nearest_distance = find_nearest_point(k_centers[0: r], points[i])
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
        print k_centers
    elif initialization_method == GONZALES_ALGORITHM:
        pass
    else:
        return False
    
    
    # clustering
    

if __name__ == "__main__":
    k_means(c2, 3, K_MEANS_PLUS_PLUS)
