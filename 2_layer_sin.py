import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


num_nodes = 400

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 2])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    weights1 = tf.Variable(tf.truncated_normal(shape=[2, num_nodes]))
    weights2 = tf.Variable(tf.truncated_normal(shape=[num_nodes, num_nodes]))
    weights3 = tf.Variable(tf.truncated_normal(shape=[num_nodes, 1]))
    biases1 = tf.Variable(tf.zeros(num_nodes))
    biases2 = tf.Variable(tf.zeros(num_nodes))
    biases3 = tf.Variable(tf.zeros(1))

    layer_1 = tf.nn.sigmoid(tf.matmul(x, weights1) + biases1)
    layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights2) + biases2)
    result = tf.nn.sigmoid(tf.matmul(layer_2, weights3) + biases3)
    
    loss = tf.nn.l2_loss(result - y)

    optimizer = tf.train.AdamOptimizer().minimize(loss)
    #global_step = tf.Variable(0)  # count the number of steps taken.
    #learning_rate = tf.train.exponential_decay(0.005, global_step, 100000, 0.96, staircase=True)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    for i in range(20000):
        tpoints = np.random.rand(100) * 20
        omegapoints = np.random.rand(100) * 2
        xpoints = np.array([tpoints, omegapoints]).T
        
        ypoints = np.sin(omegapoints*tpoints)/2+0.5
        
        tpoints_eval = np.linspace(0,40,200)
        omegapoints_eval = 0.7*np.ones(tpoints_eval.shape)
        xpoints_eval = np.array([tpoints_eval, omegapoints_eval]).T

        _, loss_result = session.run([optimizer, loss], feed_dict={x: xpoints, y: ypoints[:, None]})

        if (i % 500 == 0):
            print('Step {}, loss={}'.format(i, loss_result))
    
    out = session.run(result, feed_dict={x: xpoints_eval})
    plt.plot(tpoints_eval, out, tpoints_eval, np.sin(omegapoints_eval*tpoints_eval)/2+0.5)
    plt.show()


