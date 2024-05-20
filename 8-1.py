import tensorflow as tf
import numpy as np

vocab_size = 100
embedding_size = 25

inputs = tf.placeholder(tf.int32, shape=[None])

embedding = tf.Variable(tf.random_normal([vocab_size, embedding_size]), dtype=tf.float32)

embedded_inputs = tf.nn.embedding_lookup(embedding, inputs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_data = np.array([7])
print("input data before Embedding : ")
print(sess.run(tf.one_hot(input_data, vocab_size)))
print(tf.one_hot(input_data, vocab_size).shape)

print("Embedding : ")
print(sess.run([embedded_inputs], feed_dict={inputs : input_data}))
print(sess.run([embedded_inputs], feed_dict={inputs : input_data})[0].shape)

input_data = np.array([7, 11, 67, 42, 21])
print("input data before Embedding : ")
print(sess.run(tf.one_hot(input_data, vocab_size)))
print(tf.one_hot(input_data, vocab_size).shape)

print("Embedding : ")
print(sess.run([embedded_inputs], feed_dict={inputs : input_data}))
print(sess.run([embedded_inputs], feed_dict={inputs : input_data})[0].shape)