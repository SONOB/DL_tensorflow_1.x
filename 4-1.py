import tensorflow.compat.v1 as tf
from tensorflow import keras
import numpy as np 

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

w = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))

logits = tf.matmul(x, w) + b
y_pred = tf.nn.softmax(logits)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 100

for i in range(1000):
    indices = np.random.randint(x_train.shape[0], size=batch_size)
    batch_xs, batch_ys = x_train[indices], y_train[indices]
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test})

print("Test Accuracy: %f" % test_accuracy)

sess.close()