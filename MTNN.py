import numpy as np
import threading, pickle, time
import mnist_dataloader

# load data set
training_data, validation_data, test_data = mnist_dataloader.load_data_wrapper()

start_time = time.time()

# compute length of each instance in training_data
training_data_lengths = [np.linalg.norm(training_instance[0]) for training_instance in training_data]

# compute the length of each instance in test_data
test_data_lengths = [np.linalg.norm(test_instance[0]) for test_instance in test_data]

# for i in range(1, 100):
#     print test_data_length[i]

# classify test_data

classified_results = np.zeros(len(test_data))

# speed using multiple threads
threads = []
NUMBER_OF_THREADS = 4

def find_nearest_instance_thread(test_instance_start_index, test_instance_end_index):
    
    print test_instance_start_index, test_instance_end_index
    
    for test_instance_index in range(test_instance_start_index, test_instance_end_index):
        
        # find the nearest training instance with cosine similarity
        maximal_cosine_similarity = -1
        maximal_cosine_similarity_index = 0
        for training_instance, training_instance_index in zip(training_data, range(len(training_data))):
            # compute the cosine similarity
            # first, compute the inner product
            inner_product = np.inner(test_data[test_instance_index][0].reshape(-1), training_instance[0].reshape(-1))
            normalized_inner_product = inner_product / test_data_lengths[test_instance_index] / training_data_lengths[training_instance_index]
            
            if normalized_inner_product > maximal_cosine_similarity:
                maximal_cosine_similarity = normalized_inner_product
                maximal_cosine_similarity_index = training_instance_index
        
        classified_results[test_instance_index] = maximal_cosine_similarity_index
    

test_data_subdivisions = range(0, 1000 + 1, int(1000 / NUMBER_OF_THREADS))
test_data_subdivisions[-1] = 1000
# test_data_subdivisions = range(0, len(test_data) + 1, int(len(test_data) / NUMBER_OF_THREADS))
# test_data_subdivisions[-1] = len(test_data) - 1
# print test_data_subdivisions


for thread_index in range(NUMBER_OF_THREADS):
    thread = threading.Thread(target = find_nearest_instance_thread, args = [test_data_subdivisions[thread_index], test_data_subdivisions[thread_index + 1]])
    
    thread.start()
    threads.append(thread) 

        
# to wait until all three functions are finished
print "Waiting..."
for thread in threads:
    thread.join()
print "Complete."

# for label in classified_results:
#     print label

print "--- %s seconds ---" % (time.time() - start_time)

# for test_instance_index, nearest_instance_index in zip(range(100), classified_results):
#     print training_data[int(nearest_instance_index)][1], test_data[test_instance_index][1]

# print classified_results

# with open("classified_results.pkl", "w") as f:
#     pickle.dump(classified_results, f)
# f.close()