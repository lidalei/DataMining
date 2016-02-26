import numpy as np
import multiprocessing, time, mnist_dataloader

# load data set
training_data, validation_data, test_data = mnist_dataloader.load_data()
training_data_instances = training_data[0]
training_data_labels = training_data[1]
test_data_instances = test_data[0]
test_data_labels = test_data[1]

start_time = time.time()
# compute the length of each instance in training_data
training_data_lengths = [np.linalg.norm(training_instance) for training_instance in\
                          training_data_instances]
# compute the length of each instance in test_data
test_data_lengths = [np.linalg.norm(test_instance) for test_instance in test_data_instances]
# classify test_data
def find_nearest_instance_subprocess(test_instance_start_index, test_instance_end_index,\
                                      classified_results):
    # print test_instance_start_index, test_instance_end_index
    for test_instance_index in range(test_instance_start_index, test_instance_end_index):
        # find the nearest training instance with cosine similarity
        maximal_cosine_similarity = -1.0
        maximal_cosine_similarity_index = 0
        for training_instance, training_instance_index in\
         zip(training_data_instances, range(len(training_data_instances))):
            # compute the cosine similarity
            # first, compute the inner product
            inner_product = np.inner(test_data_instances[test_instance_index], training_instance)
            # second, normalize the inner product
            normalized_inner_product = inner_product / test_data_lengths[test_instance_index]\
             / training_data_lengths[training_instance_index]
            if normalized_inner_product > maximal_cosine_similarity:
                maximal_cosine_similarity = normalized_inner_product
                maximal_cosine_similarity_index = training_instance_index
        classified_results[test_instance_index] =\
         training_data_labels[int(maximal_cosine_similarity_index)]

if __name__ == '__main__':
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