import tensorflow as tf
tf.executing_eagerly()

x = tf.Variable(10, name="x")
y = tf.Variable(20, name="y")
z = x**2 + y**3 + 10000
print(z)
