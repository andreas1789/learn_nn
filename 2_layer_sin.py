import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_nodes = 400

plt.axis([0, 40, 0, 1])
plt.ion()

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    weights1 = tf.Variable(tf.truncated_normal(shape=[3, num_nodes]))
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

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    for i in range(60000):
        tpoints = np.random.rand(100) * 30
        omegapoints = np.random.rand(100) * 2
        taupoints = np.random.rand(100) * 1 
        xpoints = np.array([tpoints, omegapoints, taupoints]).T
        
        ypoints = np.exp(-taupoints * tpoints) * ( np.sin(omegapoints*tpoints) + 0.2*np.sin(6*omegapoints*tpoints) )/2+0.5
        
        tpoints_eval = np.linspace(0,40,400)
        omegapoints_eval = 0.7*np.ones(tpoints_eval.shape)
        taupoints_eval = 0.1*np.ones(tpoints_eval.shape)
        xpoints_eval = np.array([tpoints_eval, omegapoints_eval, taupoints_eval]).T

        _, loss_result = session.run([optimizer, loss], feed_dict={x: xpoints, y: ypoints[:, None]})

        if (i % 500 == 0):
            print('Step {}, loss={}'.format(i, loss_result))
            plt.cla()
            out = session.run(result, feed_dict={x: xpoints_eval})
            plt.plot(tpoints_eval, out, tpoints_eval, np.exp(-taupoints_eval * tpoints_eval) * ( np.sin(omegapoints_eval*tpoints_eval) + 0.2*np.sin(6*omegapoints_eval*tpoints_eval) )/2+0.5)
            plt.pause(0.05)


