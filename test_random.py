import tensorflow as tf

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

for i in range(2):
    #seed=(1,1)
    seed = tf.compat.v1.set_random_seed(0)
    var = tf.Variable(tf.random_normal([1, 1], 0.0, 0.01)) 
    print(var.eval())

