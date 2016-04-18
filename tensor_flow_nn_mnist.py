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
from SGDDataset import SGDDataSet

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 50, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 600, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

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


def inference(images, hidden1_units):
    """Build the MNIST model up to where it may be used for inference.
    
    Args:
      images: Images placeholder, from inputs().
      hidden1_units: Size of the first hidden layer.    
    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        '''
        A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations.
        It can be used and even modified by the computation. For machine learning applications,
        one generally has the model parameters be Variables.
        '''  
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=1.0 / np.sqrt(float(IMAGE_PIXELS))), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, NUM_CLASSES], stddev=1.0 / np.sqrt(float(hidden1_units))), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden1, weights) + biases
        
        return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.    
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].
    
    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name = 'xentropy')
    loss = tf.reduce_mean(cross_entropy, name = 'xentropy_mean')
    return loss
    

def training(loss, learning_rate):
    """Sets up the training Ops.
    
    Creates a summarizer to track the loss over time in TensorBoard.
    
    Creates an optimizer and applies the gradients to all trainable variables.
    
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
    
    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step = global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES).
    
    Returns:
      A scalar float32 tensor with the rate of examples (out of batch_size) that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def run(X_train, y_train, train_data, X_test, y_test):
    it_counts, loss_values, train_scores, test_scores = [], [], [], []
    
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        '''
        A placeholder, a value that we'll input when we ask TensorFlow to run a computation.
        Here None means that a dimension can be of any length.
        '''
        images_placeholder = tf.placeholder(tf.float32, shape = (None, IMAGE_PIXELS))
        labels_placeholder = tf.placeholder(tf.int32, shape = (None))
        # Build a Graph that computes predictions from the inference model.
        logits = inference(images_placeholder, FLAGS.hidden1)
        # Add to the Graph the Ops for loss calculation.
        loss_op = loss(logits, labels_placeholder)
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss_op, FLAGS.learning_rate)
        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(logits, labels_placeholder)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        
        sess = tf.Session()
        # add an operation to initialize the variables we created
        init = tf.initialize_all_variables()
        sess.run(init)
        
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
        
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            batch_xs, batch_ys = train_data.next_batch(FLAGS.batch_size)
            
            feed_dict = {images_placeholder: batch_xs, labels_placeholder: batch_ys}
            _, loss_value  = sess.run([train_op, loss_op], feed_dict = feed_dict)
            duration = time.time() - start_time
            
            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                '''
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict = feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                '''
                
                it_counts.append(step)
                loss_values.append(float(loss_value)) # to json serializable
                
                train_score = sess.run(eval_correct, feed_dict={images_placeholder: X_train, labels_placeholder: y_train})
                train_scores.append(float(train_score))
                
                test_score = sess.run(eval_correct, feed_dict={images_placeholder: X_test, labels_placeholder: y_test})
                
                test_scores.append(float(test_score))

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.train_dir, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval: {}'.format(train_score)) 
                # Evaluate against the test set.
                print('Test Data Eval: {}'.format(test_score))
        
        return it_counts, loss_values, train_scores, test_scores


if __name__ == '__main__':
    ## get dataset - MNIST
    X, y, attribute_names, target_attribute_names = get_dataset(554)
    
    '''
    # vectorize y
    vec_y = np.zeros((y.shape[0], 10), dtype = np.int32)
    for vec_y_i, y_i in zip(vec_y, y):
        vec_y_i[y_i] = 1
    '''
    ## 60,000 as training data, 10,000 as test data
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    train_data = SGDDataSet(X_train, y_train, dtype = tf.float32)
    
    for learning_rate in [0.0001, 0.001, 0.01, 0.1]:
        FLAGS.learning_rate = learning_rate
        it_counts, loss_values, train_scores, test_scores = run(X_train, y_train, train_data, X_test, y_test)
        ## save train process, iterative counts and corresponding train error, test error and loss
        train_process = {
                         'it_counts': it_counts,
                         'loss_values': loss_values,
                         'train_scores': train_scores,
                         'test_scores': test_scores
                         }
        with open('train_process_learning_rate_' + str(learning_rate) + '.json', 'w+') as f:
            json.dump(train_process, f)
        f.close()