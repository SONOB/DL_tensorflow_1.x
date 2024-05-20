import tensorflow.compat.v1 as tf

w = tf.Variable(tf.random_normal(shape=[1]), name='w')
b = tf.Variable(tf.random_normal(shape=[1]), name='b')
x = tf.placeholder(tf.float32, name="x")

linear_model = w*x + b

y = tf.placeholder(tf.float32, name="y")

loss = tf.reduce_mean(tf.square(linear_model - y))
tf.summary.scalar('loss', loss)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.FileWriter('./tensorboard_log', sess.graph)

for i in range(1000):
    sess.run(train_step, feed_dict={x:x_train, y:y_train})

    summary = sess.run(merged, feed_dict={x:x_train, y:y_train})
    tensorboard_writer.add_summary(summary, i)

x_test = [3.5, 5, 5.5, 6]

print(sess.run(linear_model, feed_dict={x:x_test}))

sess.close()