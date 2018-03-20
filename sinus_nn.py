import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

graph = tf.Graph()
num_prm = 3
num_data = 30000

"""
                A*sin(omega*t)
{A, omega, t} --------------------> y

"""

# create dataset from random with knowledge of model
train_dataset = np.random.randn(num_data, num_prm).astype(np.float32)
train_meas = np.zeros(num_data).astype(np.float32)
for d in range(0, num_data):
    A = train_dataset[d, 0]
    omega = train_dataset[d, 1]
    t = train_dataset[d,2]
    y = A * np.sin(10*omega * t)
    train_meas[d] = y

test_dataset = train_dataset[int(0.8*num_data):,:]
test_meas = train_meas[int(0.8*num_data):]


t = np.linspace(0,20,50).astype(np.float32)
eval_dataset = np.array([np.full([50],0.1).astype(np.float32), np.full([50], 0.2).astype(np.float32), t])
eval_dataset = eval_dataset.transpose()
eval_meas = 0.1*np.sin(10*0.2*t)


def accuracy(predictions, meas):
    return 100*sum(abs(predictions - meas)) / meas.shape
    

# 5-layer DNN with regularization, dropout and SGD
import math
batch_size = 512
beta = 0.01
num_nodes1 = 64
num_nodes2 = 64
num_nodes3 = 64
num_nodes4 = 64

graph = tf.Graph()

with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_prm))
    tf_train_meas = tf.placeholder(tf.float32, shape=(batch_size))
    tf_test_dataset = tf.constant(test_dataset)
    tf_eval_dataset = tf.constant(eval_dataset)

    weights1 = tf.Variable(tf.truncated_normal([num_prm, num_nodes1], stddev=math.sqrt(2.0/num_nodes1)))
    biases1 = tf.Variable(tf.zeros([num_nodes1]))
    weights2 = tf.Variable(tf.truncated_normal([num_nodes1, num_nodes2], stddev=math.sqrt(2.0/num_nodes2)))
    biases2 = tf.Variable(tf.zeros([num_nodes2]))
    weights3 = tf.Variable(tf.truncated_normal([num_nodes2, num_nodes3], stddev=math.sqrt(2.0/num_nodes3)))
    biases3 = tf.Variable(tf.zeros([num_nodes3]))
    weights4 = tf.Variable(tf.truncated_normal([num_nodes3, num_nodes4], stddev=math.sqrt(2.0/num_nodes4)))
    biases4 = tf.Variable(tf.zeros([num_nodes4]))
    weights5 = tf.Variable(tf.truncated_normal([num_nodes4, 1], stddev=math.sqrt(2.0/1)))
    biases5 = tf.Variable(tf.zeros([1]))
    
    keep_prob = tf.placeholder("float")
        
    sigmoid_layer1 = tf.nn.sigmoid(tf.matmul(tf_train_dataset, weights1) + biases1)
    logits1 = tf.nn.dropout(sigmoid_layer1, keep_prob)
    
    sigmoid_layer2 = tf.nn.sigmoid(tf.matmul(logits1, weights2) + biases2)
    logits2 = tf.nn.dropout(sigmoid_layer2, keep_prob)
    
    sigmoid_layer3 = tf.nn.sigmoid(tf.matmul(logits2, weights3) + biases3)
    logits3 = tf.nn.dropout(sigmoid_layer3, keep_prob)
    
    sigmoid_layer4 = tf.nn.sigmoid(tf.matmul(logits3, weights4) + biases4)
    logits4 = tf.nn.dropout(sigmoid_layer4, keep_prob)
    
    logits = tf.nn.sigmoid(tf.matmul(logits4, weights5) + biases5)
    
    regularizer = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + \
                  tf.nn.l2_loss(weights3) + tf.nn.l2_loss(weights4) + \
                  tf.nn.l2_loss(weights5)
        
    loss = tf.reduce_mean((logits-tf_train_meas)*(logits-tf_train_meas))
    loss = tf.reduce_mean(loss + beta * regularizer)
    
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.01, global_step, 100000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    train_prediction = logits
    
    test_layer1 = tf.nn.sigmoid(tf.matmul(tf_test_dataset, weights1) + biases1)
    test_layer2 = tf.nn.sigmoid(tf.matmul(test_layer1, weights2) + biases2)
    test_layer3 = tf.nn.sigmoid(tf.matmul(test_layer2, weights3) + biases3)
    test_layer4 = tf.nn.sigmoid(tf.matmul(test_layer3, weights4) + biases4)
    test_prediction = tf.matmul(test_layer4, weights5) + biases5

    # DNN
num_steps = 20001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")

    for step in range(num_steps):
        offset = (step * batch_size) % (train_meas.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset+batch_size),:]
        batch_meas = train_meas[offset:(offset+batch_size)]
        
        feed_dict = {tf_train_dataset : batch_data, tf_train_meas : batch_meas, keep_prob : 0.5}
        
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions.squeeze(), batch_meas))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval().squeeze(), test_meas))
    
    eval_leayer1 = tf.nn.sigmoid(tf.matmul(tf_eval_dataset, weights1) + biases1)
    eval_leayer2 = tf.nn.sigmoid(tf.matmul(eval_leayer1, weights2) + biases2)
    eval_leayer3 = tf.nn.sigmoid(tf.matmul(eval_leayer2, weights3) + biases3)
    eval_leayer4 = tf.nn.sigmoid(tf.matmul(eval_leayer3, weights4) + biases4)
    eval_prediction = tf.matmul(eval_leayer4, weights5) + biases5
    print(eval_prediction.eval())
    plt.plot(t, eval_meas, t, eval_prediction.eval().squeeze())
    plt.show()

    


