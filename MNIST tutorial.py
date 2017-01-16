# see https://www.tensorflow.org/tutorials/mnist/beginners/

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def main(_):
    # dwnload MNIST dataset from Yann LeCun's website
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # define the model (softmax regression)

    # this represents the input images
    x = tf.placeholder(tf.float32, [None, 784])
    # the weights and biases are defined as variables
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # this is the actual regression model
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # train the model
    # use cross-entropy as cost function see http://colah.github.io/posts/2015-09-Visual-Information/
    # These are the correct (on-hot vector) answers that is compared with the model predictions
    y_ = tf.placeholder(tf.float32, [None, 10])
    # The mean cross-entropy accross all training examples 
    # The raw formulation of cross-entropy,
      #
      #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
      #                                 reduction_indices=[1]))
      #
      # can be numerically unstable.
      #
      # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
      # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # use gradient descent to train the network
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # initialise the session   
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for ii in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        if (ii % 1000 == 0):
            print('batch ' + str(ii/1000)) 
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Mean accruacy accross all images: ' + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Q:\Projects\Source\temp\MNIST',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

