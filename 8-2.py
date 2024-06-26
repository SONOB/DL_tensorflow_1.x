import tensorflow as tf

w = tf.Variable(tf.random_normal(shape=[1]), name="w")
b = tf.Variable(tf.random_normal(shape=[1]), name="b")
x = tf.placeholder(tf.float32, name='x')

linear_model = w*x+b

y = tf.placeholder(tf.float32, name='y')

loss = tf.reduce_mean(tf.square(linear_model - y))

grad_clip = 5
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.apply_gradients(zip(grads, tvars))

x_train = [1,2,3,4]
y_train = [2,4,6,8]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_step, feed_dict={x:x_train, y:y_train})
    
x_test = [3.5, 5, 5.5, 6]

print(sess.run(linear_model, feed_dict={x:x_test}))

sess.close()