#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Jidan Zhong 
# 2017- Jan-31

### Linear regression
import io
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt 
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10,6)

def save_plot():
    """Save a pyplot plot to buffer."""
    # plt.figure()
    # plt.plot(x, y)
    # plt.title("test")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    # Add image summary
    summary_op = tf.image_summary("plot", image)
    # return buf
    return summary_op

def variable_summaries(var):                ###################################################################ADDDDDED########
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


## load data and explore the data

# PierceCricketData.csv saved data from http://mathbits.com/MathBits/TISection/Statistics2/linearREAL.htm

df = pd.read_csv("resources/PierceCricketData.csv",header = None)
df.head()
x_data, y_data = (df[0], df[1])
# plot data to explore
plt.figure()
plt.plot(x_data, y_data, 'ro')
plt.xlabel("# Chirps per 15 sec")
plt.ylabel("Temp in Farenhiet")
# plt.show()

 
# normalize the data so the performance is better 
x_data_n, y_data_n = ((x_data - x_data.mean())/x_data.std(),(y_data - y_data.mean())/y_data.std() )
# plot data to explore
plt.figure()
plt.plot(x_data, y_data, 'ro')
plt.xlabel("# Chirps per 15 sec")
plt.ylabel("Temp in Farenhiet")


with tf.Graph().as_default():
    with tf.name_scope('input'):

        ## preparing linear model
        Xsize= x_data.size
        Ysize= y_data.size
        X = tf.placeholder(tf.float32, shape=(Xsize), name='x-input')
        Y = tf.placeholder(tf.float32,shape=(Ysize), name='y-input')

    with tf.name_scope('model'):
        with tf.name_scope('weights'):
            W = tf.Variable(3.0, name='weight') # weight
            tf.summary.scalar('weights', W)
            # variable_summaries(W)
        with tf.name_scope('bias'):    
            B = tf.Variable(1.0, name = 'bias') # bias
            tf.summary.scalar('bias', B)
            # variable_summaries(B)
        with tf.name_scope('linear_model'):
            # construct a model
            y = X * W + B  # or # Y = tf.add(tf.mul(W, X), B)  

    with tf.name_scope('loss'):
        # seting up the loss function
        loss = tf.reduce_mean(tf.squared_difference(y,Y))   # This is wrong: loss = tf.reduce_mean(tf.square(y-Y)) 
        #tf.summary.scalar('loss', loss)
        tf.scalar_summary('loss', loss)
        # variable_summaries(loss)
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()    
    # saver = tf.train.Saver()

    # start = time.time()

    init = tf.global_variables_initializer()

    #pred = sess.run(y, feed_dict={X:x_data})

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess: # the first one means if the device doesnt exist, it can automatically appoint an existing device; 2nd means it will show the log infor for parameters and operations are on which device
        
        train_writer = tf.summary.FileWriter('/home/jidan/test/train', sess.graph)

        sess.run(init)

        convergenceTolerance = 0.0001
        previous_w = np.inf
        previous_b = np.inf

        steps = {}
        steps['w'] = []
        steps['b'] = []

        losses=[]

        for k in range(1000):
            yPred, _, weight, bias, ls, summary = sess.run([y,optimizer,W,B,loss, merged], feed_dict={X:np.asarray(x_data_n), Y:np.asarray(y_data_n)})
            train_writer.add_summary(summary, k)
            print "step: %d, loss: %f" %(k,ls)
          
            steps['w'].append(weight)
            steps['b'].append(bias)
            losses.append(ls)
            if (np.abs(previous_w - weight) or np.abs(previous_b - bias) ) <= convergenceTolerance :
                print "Finished by Convergence Criterion"
                print k
                print ls
                break
            previous_w = weight 
            previous_b = bias
        print "In the model, Final W: %f, Final B: %f" %(weight, bias)
        # model without normalizing data will be :
        # y = X * (W * y_data.std() / x_data.std()) + (y_data.std() * b  + y_data.mean() - x_data.mean() * W * y_data.std() / x_data.std())
        print "W for original data: %f, B for original data: %f" %(weight* y_data.std()/ x_data.std(), y_data.std() * bias  + y_data.mean() - x_data.mean() * weight * y_data.std() / x_data.std())    
        # final step show the graph 

        plt.figure()
        plt.plot(x_data_n, yPred)
        plt.plot(x_data_n, y_data_n, 'ro')
        plt.xlabel("# Chirps per 15 sec normalized ")
        plt.ylabel("Temp in Farenhiet normalized")
        # save the figure to buffer
        summary_op = save_plot()
        summary1 = sess.run(summary_op)
        train_writer.add_summary(summary1)


        ########## Plot the figures for self exploration
        plt.figure(1)
        plt.subplot(221)

        plt.plot(x_data_n, yPred)
        plt.plot(x_data_n, y_data_n, 'ro')
        # label the axis
        plt.xlabel("# Chirps per 15 sec normalized")
        plt.ylabel("Temp in Farenhiet normalized")
      #  plt.show()

        # print the loss function 
        plt.subplot(223)
        plt.plot(range(k+1), losses)
        plt.xlabel("step")
        plt.ylabel("loss")
      #  plt.show()



        plt.subplot(224)
        plt.plot(x_data, yPred * y_data.std() + y_data.mean() ) 
        plt.plot(x_data, y_data, 'ro')
        # label the axis
        plt.xlabel("# Chirps per 15 sec")
        plt.ylabel("Temp in Farenhiet")


      #  plt.show()
        # print the step changes 
        plt.subplot(222)

        converter = plt.colors
        cr, cg, cb = (0.0, 1.0, 0.0)
        for f in range(k):
            cb +=1.0 / k
            cg -=1.0 / k
            cr +=1.0 / k / 2            #############3
            if cb > 1.0: cb = 1.0
            if cg < 0.0: cg = 0.0
            if cr > 1.0: cr = 0.5
            a = steps['w'][f]
            b = steps['b'][f]
            f_y =  np.vectorize(lambda x: a * x + b) (x_data_n)
            line = plt.plot(x_data_n, f_y)
            plt.setp(line, color=(cr,cg,cb))
            plt.plot(x_data_n, y_data_n,'ro')
            plt.xlabel("# Chirps per 15 sec normalized")
            plt.ylabel("Temp in Farenhiet normalized")
        plt.show()
