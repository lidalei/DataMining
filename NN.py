import numpy as np
import mnist_dataloader, time

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

classified_results = []

for test_instance, test_instance_index in zip(test_data, range(len(test_data))):
    # find the nearest training instance with cosine similarity
    maximal_cosine_similarity = -1
    maximal_cosine_similarity_index = 0
    for training_instance, training_instance_index in zip(training_data, range(len(training_data))):
        # compute the cosine similarity
        # first, compute the inner product
        inner_product = np.inner(test_instance[0].reshape(-1), training_instance[0].reshape(-1))
        normalized_inner_product = inner_product / test_data_lengths[test_instance_index] / training_data_lengths[training_instance_index]
        
        if normalized_inner_product > maximal_cosine_similarity:
            maximal_cosine_similarity = normalized_inner_product
            maximal_cosine_similarity_index = training_instance_index
    
    classified_results.append([maximal_cosine_similarity_index, maximal_cosine_similarity])
    
print "--- %s seconds ---" % (time.time() - start_time)

result = []

print classified_results

# with open("classified_results.pkl", "w") as f:
#     pickle.dump(classified_results, f)
# f.close()