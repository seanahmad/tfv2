import tensorflow as tf
x = tf.Variable(10, name="x")
y = tf.Variable(20, name="y")
z = x**2 + y**3 + 10000

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    answer = z.eval()

answer
