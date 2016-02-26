import pickle
import numpy as np

with open("classified_results.pkl", "r") as f:
    result = pickle.load(f)
f.close()

count = 0
confusion_matrix = np.zeros((10, 10), dtype=np.int)

for labels in result:
    if labels[0] == labels[1]:
        count += 1
        
    confusion_matrix[labels[0]][labels[1]] += 1
    
        
print "Accuracy is ", count, "/", len(result)

print confusion_matrix