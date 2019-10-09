import tensorflow as tf
x = tf.Variable(10.0)
y = tf.Variable(20.0)
z = x**2 + y**3 + 10000

with tf.compat.v1.Session() as sess:
    answer = z.numpy()

answer
