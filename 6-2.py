import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.02
training_epochs = 50
batch_size = 256
display_step = 1
examples_to_show = 10
input_size = 784
hidden1_size = 256
hidden2_size = 128

x = tf.placeholder(tf.float32, shape=[None, input_size])

def build_autoencoder(x):

  W1 = tf.Variable(tf.random_normal(shape=[input_size, hidden1_size]))
  b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
  H1_output = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
  
  W2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]))
  b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]))
  H2_output = tf.nn.sigmoid(tf.matmul(H1_output,W2) + b2)
  
  W3 = tf.Variable(tf.random_normal(shape=[hidden2_size, hidden1_size]))
  b3 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
  H3_output = tf.nn.sigmoid(tf.matmul(H2_output,W3) + b3)

  W4 = tf.Variable(tf.random_normal(shape=[hidden1_size, input_size]))
  b4 = tf.Variable(tf.random_normal(shape=[input_size]))
  reconstructed_x = tf.nn.sigmoid(tf.matmul(H3_output,W4) + b4)

  return reconstructed_x

y_pred = build_autoencoder(x)

y_true = x

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
  
  sess.run(tf.global_variables_initializer())

  
  for epoch in range(training_epochs):
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      _, current_loss = sess.run([train_step, loss], feed_dict={x: batch_xs})
    if epoch % display_step == 0:
      print("반복(Epoch): %d, 손실 함수(Loss): %f" % ((epoch+1), current_loss))

  reconstructed_result = sess.run(y_pred, feed_dict={x: mnist.test.images[:examples_to_show]})

  f, a = plt.subplots(2, 10, figsize=(10, 2))
  for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(reconstructed_result[i], (28, 28)))
  f.savefig('reconstructed_mnist_image.png')
  f.show()
  plt.draw()
  plt.waitforbuttonpress()