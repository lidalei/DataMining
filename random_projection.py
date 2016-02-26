"""
Random projection, Assignment 1c
"""
import numpy as np
import matplotlib.pylab as plt
import random, mnist_dataloader
from numpy import dtype

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

if __name__ == "__main__":
    # load data set
    training_data, validation_data, test_data = mnist_dataloader.load_data()
    # row vector (matrix)
    training_data_instances = training_data[0]
    training_data_labels = training_data[1]
    # row vector (matrix)
    test_data_instances = test_data[0]
    test_data_labels = test_data[1]
    # dimension of a training data instance
    d = training_data_instances.shape[1]
        
    for k in [50, 100, 500]:
        random_projection_matrix =  generate_random_projection_matrix(k, d)
        # transpose to column vector (matrix) before projection and recover after projection
        random_projected_matrix = np.transpose(random_projection(random_projection_matrix, np.transpose(training_data_instances[0:20])))
        
        print random_projected_matrix[0], random_projected_matrix.shape
        