# see https://www.tensorflow.org/tutorials/mnist/beginners/

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# Helper functions to make bias and weights

# This creates weight variables that have a small amount of noise to prevent 0 gradients
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
# This creates bias factors that are slightly positive to prevent the "dead neuron" problem
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 2-D convolution of x with W using stride of one for all dimensions and zero-padding at borders
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# performs max-pooling on a 2x2 window using zeo-padding on the borders
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def main(_):
    # dwnload MNIST dataset from Yann LeCun's website
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # define the model (multilayer convnet)

    # this represents the input images
    x = tf.placeholder(tf.float32, [None, 784])
    # These are the correct (on-hot vector) answers that is compared with the model predictions
    y_ = tf.placeholder(tf.float32, [None, 10])
    # the first layer of the convnet
    
    # the first weight layer will compute 32 features for each 5x5 patch (the filter of the 2d convolution)
    W_conv1 = weight_variable([5, 5, 1, 32])
    # bias variables to add to each of the 32 dimensions
    b_conv1 = bias_variable([32])
    # reshape the image tensor into a fromat that can be convolved 
    x_image = tf.reshape(x, [-1,28,28,1])
    # convolve, add bias and apply rectified linear unit (ReLU neuron) (max(0,x)) to result
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # max pooling in 2x2 windows reduce the image size to 14x14 from 28x28
    h_pool1 = max_pool_2x2(h_conv1)

    # the second layer of the convnet
    # this time we have 64 features for each 5x5 patch (why 32 and then 64?)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # max pooling will result in 7x7 image this time
    h_pool2 = max_pool_2x2(h_conv2)

    # densly connected layer with 1024 neurons
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # add dropout before the readout layer to prevent overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # train the model
    # use cross-entropy as cost function see http://colah.github.io/posts/2015-09-Visual-Information/
    
    # The mean cross-entropy accross all training examples 
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    # use the more sophisticated ADAM optimiser
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # initialise the session   
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    sess.run(tf.global_variables_initializer())
    # training loop
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Q:\Projects\Source\temp\MNIST',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

