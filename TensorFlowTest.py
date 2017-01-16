import tensorflow as tf

hello = tf.constant('Hello, Tensorflow on Windows!')

sess = tf.Session()
print(sess.run(hello))