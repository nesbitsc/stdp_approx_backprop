"""
Continuous version of EP. Rules 1 and 2 are as in the paper
Generalization of Equilibrium Propagation to Vector Field Dynamics.
"""

import argparse
import csv
import tensorflow as tf
import numpy as np
import os, sys
# from params import *
sys.path.insert(0,"../../Classes")
sys.path.insert(0,"../")
from layers import *
from util import *

# np.random.seed(1111)
# tf.random.set_seed(1111)

########################
# Command line arguments
########################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains an MNIST classifier using various forms of Equilibrium Propagation')
    parser.add_argument('--beta0', type=float, default=-0.1, help='set beta_0 learning rule parameter')
    parser.add_argument('--beta1', type=float, default=0.5, help='set beta_1 learning rule parameter')
    parser.add_argument('--phase_1_n_steps', type=int, default=40, help='steps in phase 1 of training')
    parser.add_argument('--phase_2_n_steps', type=int, default=20, help='steps in phase 2 of training')
    parser.add_argument('--batch_size', type=int, default=20, help='size of each minibatch')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--input_size', type=int, default=784, help='size of input layer')
    parser.add_argument('--hidden_size', type=int, default=500, help='size of hidden layer')
    parser.add_argument('--target_size', type=int, default=10, help='size of output layer')
    parser.add_argument('--beta', type=float, default=1.0, help='set beta learning rate')
    parser.add_argument('--epsilon', type=float, default=0.05, help='set epsilon learning rate')
    parser.add_argument('--sparsity', type=float, default=0.1, help='set sparsity value')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='select learning rate')
    parser.add_argument('--use_bias', action='store_true', help='include fixed bias')
    parser.add_argument('--trace_decay', type=float, default=0.005,help='set trace decay')


    args = parser.parse_args()

    n_features = args.hidden_size

    # the below code is inefficient, but I wanted to change as little of your code as possible right now
    phase_1_n_steps = args.phase_1_n_steps
    phase_2_n_steps = args.phase_2_n_steps
    input_size = args.input_size
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    target_size = args.target_size
    beta = args.beta
    beta0 = args.beta0
    beta1 = args.beta1
    epsilon = args.epsilon
    sparsity = args.sparsity
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    use_bias=args.use_bias
    trace_decay = args.trace_decay

    ########################
    # Put existing code here
    ########################

    M = int(np.sqrt(n_features))
    # np.random.seed(1234)
    # tf.random.set_seed(1234)

    # Create output directory and log files
    #output_dir = sys.argv[1]
    output_dir = "output"
    os.system("mkdir -p "+output_dir)
    os.system("touch "+output_dir+"/run_log.txt")
    run_log = open(output_dir+"/run_log.txt","w",1)
    outf = output_dir+'/output.csv'

    ###########
    # Load data
    ###########
    data_path = "../../../datasets/"
    data = np.genfromtxt(data_path+"mnist_train.csv",delimiter=",",dtype=np.uint8) #Load MNIST data from csv file
    test_data = np.genfromtxt(data_path+"mnist_test.csv",delimiter=",",dtype=np.uint8) #Load MNIST data from csv file

    # Training set
    images = data[:,1:]/255.
    targets = np.zeros((60000,target_size))
    targets[np.arange(60000),data[:,0]]=1.0

    # Test set
    test_images = test_data[:,1:]/255.
    test_targets = np.zeros((10000,target_size))
    test_targets[np.arange(10000),test_data[:,0]]=1.0

    #####################
    # Create weight array
    #####################
    W_xh = getRandomArray((input_size,hidden_size))
    W_hy = getRandomArray((hidden_size,target_size))
    W_yh = getRandomArray((target_size,hidden_size))

    b_h = tf.random.uniform((1,hidden_size),minval=0,maxval=4./(input_size+hidden_size))
    b_y = tf.random.uniform((1,target_size),minval=0,maxval=4./hidden_size)

    #################
    # Setup variables
    #################
    h0_activity = tf.Variable(tf.zeros((batch_size,hidden_size)),dtype=tf.float32) # Local mins for h free
    y0_activity = tf.Variable(tf.zeros((batch_size,target_size)),dtype=tf.float32) # Local mins for y
    activity = tf.Variable(np.zeros(hidden_size),dtype=tf.float32) # Activity

    #########################
    # Set activation function
    #########################
    x = PoissonLayer((batch_size,input_size))
    h = PoissonLayer((batch_size,hidden_size))
    y = PoissonLayer((batch_size,target_size))
    d = PoissonLayer((batch_size,target_size))

##################################################################################################################
##################################################################################################################

    #######
    # Train
    #######
    for epoch in range(n_epochs):
        for batch in range(60000//batch_size): # Iterate over mini batches
            x.setActivity(images[batch*batch_size:(batch+1)*batch_size]) # Set input batch
            d.setActivity(targets[batch*batch_size:(batch+1)*batch_size]) # Set targets

            #########
            # Phase 1
            #########

            #Activations / firing rates
            h.updateActivity()
            y.updateActivity()

            x.updateSpike()
            h.updateSpike()
            y.updateSpike()
            d.updateSpike()

            x.resetTrace()
            h.resetTrace()
            y.resetTrace()
            d.resetTrace()
            for step in range(phase_1_n_steps):

                #Inputs
                h_input = x.spike()@W_xh + y.spike()@W_yh
                y_input = h.spike()@W_hy

                if use_bias:
                    h_input += b_h
                    y_input += b_y

                h.setInput(h_input)
                y.setInput(y_input)

                #Update potentials
                h.updateState(epsilon)
                y.updateState(epsilon)

                #Activations / firing rates
                h.updateActivity()
                y.updateActivity()

                x.updateSpike()
                h.updateSpike()
                y.updateSpike()

            # Print to log
            if batch%50==0:

                # Get Predictions
                predictions = tf.math.argmax(y.activity(),axis=1)
                actual = tf.math.argmax(d.activity(),axis=1)
                accuracy = tf.reduce_mean(tf.cast(tf.math.equal(actual,predictions),tf.float32))
                hidden_activity = np.sum(h.activity().numpy()>1e-8)/batch_size

                run_log.write("\n\nRunning batch "+str(batch)+" / Epoch "+str(epoch)+"\n\n")
                run_log.write("Accuracy: "+str(100*accuracy.numpy())+"%\n\n")
                run_log.write("Average nodes active: "+str(hidden_activity)+" out of "+str(hidden_size)+"="+str(100*hidden_activity/hidden_size)+"%\n")

                # Average nodes active
                ana = tf.reduce_mean(tf.reduce_sum(tf.cast(h.activity()>1e-25,tf.float32),[1]))
                avg_nodes_active = ana.numpy()
                run_log.write("Active nodes at time " + str(batch) + " :" + str(avg_nodes_active) + "%\n")
                weight_norms = tf.norm(W_xh).numpy(), tf.norm(W_hy).numpy(), tf.norm(W_yh).numpy()
                run_log.write("L2 norm of weights "+str(weight_norms[0])+", "+str(weight_norms[1])+", "+str(weight_norms[2]))

            #########
            # Phase 2
            #########
            for step in range(phase_2_n_steps):
                #Save local mins
                h0_activity.assign(h.activity())
                y0_activity.assign(y.activity())

                # Inputs
                h_input = x.spike()@W_xh + y.spike()@W_yh
                y_input = h.spike()@W_hy + beta*(d.spike()-y.spike())

                if use_bias:
                    h_input += b_h
                    y_input += b_y

                h.setInput( h_input )
                y.setInput( y_input )

                # # Update potentials
                h.updateState(epsilon)
                y.updateState(epsilon)

                #Activations / firing rates
                h.updateActivity()
                y.updateActivity()

                x.updateSpike()
                h.updateSpike()
                y.updateSpike()
                d.updateSpike()

                x.updateTrace(trace_decay)
                h.updateTrace(trace_decay)
                y.updateTrace(trace_decay)
                d.updateTrace(trace_decay)

                ################
                # Update weights
                ################
                dh = h.activity()-h0_activity
                dy = y.activity()-y0_activity

                # Compute weight updates
                dW_xh = (tf.transpose(x.trace())@h.spike() - tf.transpose(x.spike())@h.trace())/batch_size
                dW_hy = (tf.transpose(h.trace())@y.spike() - tf.transpose(h.spike())@y.trace())/batch_size
                dW_yh = (tf.transpose(y.trace())@h.spike() - tf.transpose(y.spike())@h.trace())/batch_size

                W_xh.assign_add(learning_rate*dW_xh)
                W_hy.assign_add(learning_rate*dW_hy)
                W_yh.assign_add(learning_rate*dW_yh)

######################################################################################################################
######################################################################################################################

        ######
        # Test
        ######
        total_test_error = 0
        total_activity_hist = np.zeros(hidden_size)

        for batch in range(10000//batch_size): # Iterate over mini batches
            x.setActivity(test_images[batch*batch_size:(batch+1)*batch_size]) # Set input batch
            d.setActivity(test_targets[batch*batch_size:(batch+1)*batch_size]) # Set targets

            #Activations / firing rates
            h.updateActivity()
            y.updateActivity()
            for step in range(phase_1_n_steps):

                #Inputs
                h.setInput( x.activity()@W_xh+y.activity()@W_yh )
                y.setInput( h.activity()@W_hy )

                #Update potentials
                h.updateState(epsilon)
                y.updateState(epsilon)

                #Activations / firing rates
                h.updateActivity()
                y.updateActivity()

            # Average nodes active
            activity_hist = tf.reduce_sum(tf.cast(h.activity()>1e-15,tf.float32),axis=(0,))
            total_activity_hist += activity_hist.numpy()

            # Get test accuracy
            predictions = tf.math.argmax(y.activity(),axis=1)
            actual = tf.math.argmax(d.activity(),axis=1)
            accuracy = tf.reduce_sum(tf.cast(tf.math.equal(actual,predictions),tf.float32))
            total_test_error += accuracy.numpy()

        total_test_error /= 10000
        total_activity_hist /= 10000

        with np.printoptions(precision=1):
            run_log.write("Activities: "+str(total_activity_hist)+"\n")
        run_log.write("Test accuracy "+str(100*total_test_error)+"%")
        print("Test accuracy ",100*total_test_error,"%")

    fields = [args.beta0,args.beta1,str(100*total_test_error)] + list(total_activity_hist)
    with open(outf, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    run_log.close()
