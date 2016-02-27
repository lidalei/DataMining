"""
Random projection, Assignment 1c
"""
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random, mnist_dataloader
from scipy.spatial.distance import euclidean
import multiprocessing
import time    

# load data set
training_data, validation_data, test_data = mnist_dataloader.load_data()
training_data_instances = training_data[0]
training_data_labels = training_data[1]
test_data_instances = test_data[0]
test_data_labels = test_data[1]

"""
Generate random projection matrix R
@param: k, the reduced number of dimensions
@param: d, the original number of dimensions
@return: R, the generated random projection matrix, k * d size 
"""

def generate_random_projection_matrix(k, d):
    R = np.zeros((k, d), dtype = np.float64)
    for r in np.nditer(R, op_flags=['readwrite']):
        r[...] = random.randint(0, 1)
        if r[...] == 0:
            r[...] = -1
    
    R *= 1.0 / np.sqrt(k)
    
    return R

"""
random projection matrix P into R
@param R: random projection matrix
@param P: matrix to be reduced in dimension
@return: Q: projected matrix of P on R
"""
def random_projection(R, P):
    if R.shape[1] != P.shape[0]:
        return False
    
    print R.shape, P.shape
    
    return np.dot(R, P)

"""
plot heat map
"""
def heat_map():
    # dimension of a training data instance
    d = training_data_instances.shape[1]
    # first m instances considered
    m = 20
    
    fig, axeses = plt.subplots(1, 3)
    fig.suptitle("Distortion of random projection", fontsize = "large")
    axeses = np.reshape(axeses, axeses.size)
    
    for k, axes in zip([50, 100, 500], axeses):
        ## generate random projection matrix
        random_projection_matrix =  generate_random_projection_matrix(k, d)
        ## random projection
        """
        # transpose to column vector (matrix) before projection and recover after projection
        random_projected_matrix = np.transpose(random_projection(random_projection_matrix, np.transpose(training_data_instances[0:20])))
        
        print random_projected_matrix[0], random_projected_matrix.shape
        """
        
        m_instances = training_data_instances[0:m]
        projected_m_instances = np.zeros((m, k), dtype = np.float64)
        for i in range(m_instances.shape[0]):
            for j in range(projected_m_instances.shape[1]):
                projected_m_instances[i][j] = np.dot(random_projection_matrix[j], m_instances[i])
        # print  projected_m_instances[0]
        ## evaluate distortion
        m_instances_distortions = np.zeros((m, m), dtype = np.float64)
        for i in range(m_instances_distortions.shape[0]):
            for j in range(i + 1, m_instances_distortions.shape[1]):
                m_instances_distortions[i][j] = euclidean(projected_m_instances[i], projected_m_instances[j]) / euclidean(m_instances[i], m_instances[j])
                m_instances_distortions[j][i] = m_instances_distortions[i][j]
        # heat map
        axes.set_title("k=" + str(k), fontsize = "medium")
        # align colormap with heatmap
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        im = axes.imshow(m_instances_distortions)
        axes.set_xlabel("instances", fontsize = "medium")
        plt.colorbar(mappable = im, ax = axes, cax = cax, orientation='vertical')
    plt.show()

"""
classify test_data
@param test_instance_start_index
@param test_instance_end_index
@param classified_results: classified results, shared by different subprocesses
@return: None. Note, the updated classified results array.
"""
def find_nearest_instance_subprocess(test_instance_start_index, test_instance_end_index,\
                                      classified_results):
    # print test_instance_start_index, test_instance_end_index
    for test_instance_index in range(test_instance_start_index, test_instance_end_index):
        # find the nearest training instance with euclidean distance
        minimal_euclidean_distance = euclidean(test_data_instances[test_instance_index], training_data_instances[0])
        minimal_euclidean_distance_index = 0
        for training_instance, training_instance_index in\
         zip(training_data_instances, range(len(training_data_instances))):
            # compute the euclidean distance
            euclidean_distance = euclidean(test_data_instances[test_instance_index], training_instance)
            if euclidean_distance < minimal_euclidean_distance:
                minimal_euclidean_distance = euclidean_distance
                minimal_euclidean_distance_index = training_instance_index
        classified_results[test_instance_index] =\
         training_data_labels[int(minimal_euclidean_distance_index)]

if __name__ == '__main__':
    ## plot heatmap
    # heat_map()
    
    start_time = time.time()
    multiprocessing.freeze_support()
    # speed using multiple processes
    NUMBER_OF_PROCESSES = 4
    processes = []
    # shared by different processes, to be mentioned is that
    # global variable is only read within processes
    # the update of global variable within a process will not be submitted 
    classified_results = multiprocessing.Array('i', len(test_data_instances), lock = False)
    test_data_subdivisions = range(0, len(test_data_instances) + 1,\
                                    int(len(test_data_instances) / NUMBER_OF_PROCESSES))
    test_data_subdivisions[-1] = len(test_data_instances)
    for process_index in range(NUMBER_OF_PROCESSES):
        process = multiprocessing.Process(target = find_nearest_instance_subprocess,\
                                           args = (test_data_subdivisions[process_index],\
                                                    test_data_subdivisions[process_index + 1],\
                                                     classified_results))
        process.start()
        processes.append(process)
        
    print "Waiting..."
    # wait until all processes are finished
    for process in processes:
        process.join()
    print "Complete."
    print "--- %s seconds ---" % (time.time() - start_time)
    
    error_count = 0
    confusion_matrix = np.zeros((10, 10), dtype=np.int)
    for test_instance_index, classified_label in zip(range(len(test_data_instances)),\
                                                      classified_results):        
        if test_data_labels[test_instance_index] != classified_label:
            error_count += 1
        confusion_matrix[test_data_labels[test_instance_index]][classified_label] += 1        
        
    print "Error rate is", 100.0 * error_count / len(test_data_instances), "%"
    print "Confusion matrix is", confusion_matrix