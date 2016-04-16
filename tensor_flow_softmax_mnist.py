# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import os, time, json
from openml.apiconnector import APIConnector
from scipy.io.arff import loadarff
import numpy as np
import matplotlib.pylab as plt

def get_dataset(did):
    home_dir = os.path.expanduser("~")
    openml_dir = os.path.join(home_dir, ".openml")
    cache_dir = os.path.join(openml_dir, "cache")
    
    with open(os.path.join(openml_dir, "apikey.txt"), 'r') as fh:
        key = fh.readline().rstrip('\n')
    fh.close()
    
    openml = APIConnector(cache_directory = cache_dir, apikey = key)
    dataset = openml.download_dataset(did)
    # print('Data-set name: %s'%dataset.name)
    # print(dataset.description)
    _, meta = loadarff(dataset.data_file)
    target_attribute = dataset.default_target_attribute
    target_attribute_names = meta[target_attribute][1]
    X, y, attribute_names = dataset.get_dataset(target = target_attribute, return_attribute_names = True)
    
    return X, y, attribute_names, target_attribute_names

if __name__ == '__main__':
    ## get dataset - MNIST
    X, y, attribute_names, target_attribute_names = get_dataset(554)
    
    # vectorize y
    vec_y = np.zeros((y.shape[0], 10), dtype = np.int32)
    for vec_y_i, y_i in zip(vec_y, y):
        vec_y_i[y_i] = 1
    
    ## 60,000 as training data, 10,000 as test data
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = vec_y[:60000], vec_y[60000:]
        
    BATCH_SIZE = 1000
    
    '''
    A placeholder, a value that we'll input when we ask TensorFlow to run a computation.
    Here None means that a dimension can be of any length.
    '''
    x = tf.placeholder(tf.float32, [None, 784])
    
    
    y_true = tf.placeholder(tf.float32, [None, 10])
    
    '''
    A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations.
    It can be used and even modified by the computation. For machine learning applications,
    one generally has the model parameters be Variables.
    '''
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    y_pred = tf.nn.softmax(tf.matmul(x, W) + b)
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]))
    
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
    
    '''
    add an operation to initialize the variables we created
    '''
    init = tf.initialize_all_variables()
    
    sess = tf.Session()
    sess.run(init)
    
    for i in range(1000):        
        batch_indices = np.random.randint(X_train.shape[0], size = BATCH_SIZE, dtype = np.int32)
        batch_xs, batch_ys = X_train[batch_indices], y_train[batch_indices]
        sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})
     
    '''
    Evaluating Our Model
    '''
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    print(sess.run(accuracy, feed_dict={x: X_test, y_true: y_test}))
    