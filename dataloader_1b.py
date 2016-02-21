import numpy as np


def load_data_1b(fpath):
    data = []
    f = open(fpath, 'r')
    for line in f:
        words = line.split()
        data.append(words)
    f.close()
    arr = np.array(data, dtype=np.float64)
    return arr[:, 1:]

if __name__ == "__main__":
    c1 = load_data_1b("./data1b/C1.txt")