"""
Continuous version of EP. Rules 1 and 2 are as in the paper
Generalization of Equilibrium Propagation to Vector Field Dynamics.
"""

import argparse
import csv
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os, sys
sys.path.insert(0,"../../Classes")
sys.path.insert(0,"../")
from layers import *
from util import *

#################################
# Pick GPU with least utilization
#################################
import GPUtil
deviceIDs = GPUtil.getAvailable(order = 'memory', limit = 4, maxLoad = 0.5,\
                                maxMemory = 0.5, includeNan=False, excludeID=[],\
                                excludeUUID=[])
os.environ["CUDA_VISIBLE_DEVICES"]=str(deviceIDs[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains an MNIST classifier using various forms of Equilibrium Propagation')
    parser.add_argument('--beta0', type=float, default=0.0, help='set beta_0 learning rule parameter')
    parser.add_argument('--beta1', type=float, default=0.5, help='set beta_1 learning rule parameter')
    parser.add_argument('--activation_type', default='hard', type=str, help='type of activation function to use')
    parser.add_argument('--norm', type=str, default='none', help='type of normalization to use')
    parser.add_argument('--symmetric', action='store_true', help='enable symmetric learning')
    parser.add_argument('--phase_1_n_steps', type=int, default=20, help='steps in phase 1 of training')
    parser.add_argument('--phase_2_n_steps', type=int, default=4, help='steps in phase 2 of training')
    parser.add_argument('--batch_size', type=int, default=20, help='size of each minibatch')
    parser.add_argument('--n_epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--input_size', type=int, default=784, help='size of input layer')
    parser.add_argument('--hidden_size', type=int, default=500, help='size of hidden layer')
    parser.add_argument('--target_size', type=int, default=10, help='size of output layer')
    parser.add_argument('--beta', type=float, default=1.0, help='set beta learning rate')
    parser.add_argument('--epsilon', type=float, default=0.5, help='set epsilon learning rate')
    parser.add_argument('--decay_rate', type=list, default=[1e-4,1e-4], help='set weight decay rate')
    parser.add_argument('--learning_rule', type=int, default=0, help='select which learning rule to use')
    parser.add_argument('--learning_rate', type=float, default=[0.1,0.05], nargs='+', help='select learning rate')
    parser.add_argument('--optimal', type = float, default=0.2, help='set desired activity level')
    parser.add_argument('--threshold', type=float, default=1e-3, help='threshold between 0 and 1 for activation function')
    parser.add_argument('--tau', type=float, default=1e3, help='decay rate constant for LIF neurons')
    parser.add_argument('--reset_val', type=float, default=0.0, help='set LIF reset voltage')
    parser.add_argument('--max_val', type=float, default=1.0, help='set maximum activity value for layer')
    parser.add_argument('--no_bias', action='store_true', help='include fixed bias')
    parser.add_argument('--weight_init', type=str, default='herman', help='choose type of weight initialization')
    parser.add_argument('--bias_init', type=str, default='nesbit', help='choose type of bias initialization')


    args = parser.parse_args()
    
    args_dict = vars(args)
    n_features = args.hidden_size
    activation_type = args.activation_type
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
    decay_rate = args.decay_rate
    activation_type = args.activation_type
    n_epochs = args.n_epochs
    symmetric = args.symmetric
    learning_rule = args.learning_rule
    learning_rate = args.learning_rate
    optimal = args.optimal
    no_bias=args.no_bias
    norm = args.norm
    max_val=args.max_val
    test_accs = []

    M = int(np.sqrt(n_features))

    # Create output directory and log files
    output_dir = "output"
    os.system("mkdir -p "+output_dir)
    os.system("touch "+output_dir+"/run_log.txt")
    run_log = open(output_dir+"/run_log.txt","a",1)
    outf = output_dir+'/output.csv'
    run_log.write("\n"+str(args_dict))

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
    if activation_type == "tanh" or activation_type == "inv_tanh":
        targets = 2*targets-1

    # Test set
    test_images = test_data[:,1:]/255.
    test_targets = np.zeros((10000,target_size))
    test_targets[np.arange(10000),test_data[:,0]]=1.0
    if activation_type == "tanh" or activation_type == "inv_tanh":
        test_targets = 2*test_targets-1

    #####################
    # Create weight array
    #####################

    if args.weight_init == 'herman':
        W_xh = getRandomArray((input_size,hidden_size))
        W_hy = getRandomArray((hidden_size,target_size))
        if not symmetric:
            W_yh = getRandomArray((target_size,hidden_size))
    elif args.weight_init == 'nesbit':
        W_xh = getRandomArray((input_size,hidden_size),mn=-4./(input_size+hidden_size),mx=4./(input_size+hidden_size))
        W_hy = getRandomArray((hidden_size,target_size),mn=-4./(target_size+hidden_size),mx=4./(target_size+hidden_size))
        if not symmetric:
            W_yh = getRandomArray((target_size,hidden_size),mn=-4./(target_size+hidden_size),mx=4./(target_size+hidden_size))
    elif args.weight_init == 'kendall':
        W_xh = getRandomArray((input_size,hidden_size),mn=0,mx=4./(input_size+hidden_size))
        W_hy = getRandomArray((hidden_size,target_size),mn=0,mx=4./(target_size+hidden_size))
        if not symmetric:
            W_yh = getRandomArray((target_size,hidden_size),mn=0,mx=4./(target_size+hidden_size))
    elif args.weight_init == 'bengio':
        W_xh = tf.Variable(tf.random.uniform((input_size,hidden_size),minval=(-tf.sqrt(6./(input_size+hidden_size))),\
                                         maxval=(tf.sqrt(6./(input_size+hidden_size)))),dtype=tf.float32)
        W_hy = tf.Variable(tf.random.uniform((hidden_size,target_size),minval=(-tf.sqrt(6./(hidden_size+target_size))),\
                                         maxval=(tf.sqrt(6./(hidden_size+target_size)))),dtype=tf.float32)
        if not symmetric:
            W_yh = tf.Variable(tf.random.uniform((target_size,hidden_size),minval=(-tf.sqrt(6./(hidden_size+target_size))),\
                                             maxval=(tf.sqrt(6./(hidden_size+target_size)))),dtype=tf.float32)
    elif args.weight_init == 'load':
        W_xh = np.load('W_xh.npy')
        W_hy = np.load('W_hy.npy')
        W_yh = np.load('W_yh.npy')
    else:
        assert(0)
    if symmetric:
        W_yh = tf.transpose(W_hy)

    # Bias initialization
    if args.bias_init == 'bengio':
        b_h = tf.random.uniform((1,hidden_size),minval=(-tf.sqrt(6./(input_size+hidden_size))),\
                                maxval=(tf.sqrt(6./(input_size+hidden_size))))
        b_y = tf.random.uniform((1,target_size),minval=(-tf.sqrt(6./(hidden_size+target_size))),\
                                maxval=(tf.sqrt(6./(hidden_size+target_size))))
    elif args.bias_init == 'kendall':
        b_h = tf.random.uniform((1,hidden_size),minval=0,maxval=4./(input_size+hidden_size))
        b_y = tf.random.uniform((1,target_size),minval=0,maxval=4./hidden_size)
    elif args.bias_init == 'nesbit':
        b_h = tf.random.uniform((1,hidden_size),minval=-4./(input_size+hidden_size),maxval=4./(input_size+hidden_size))
        b_y = tf.random.uniform((1,target_size),minval=-4./hidden_size,maxval=4./hidden_size)
    elif args.bias_init == 'herman':
        b_h = getRandomArray((1,hidden_size))
        b_y = getRandomArray((1,target_size))
    elif args.bias_init == 'load':
        b_h = np.load('b_h.npy')
        b_y = np.load('b_y.npy')


    #################
    # Set up variables
    #################
    x = tf.Variable(tf.zeros((batch_size,input_size)),dtype=tf.float32) #Input vector
    h0_state = tf.Variable(tf.zeros((batch_size,hidden_size)),dtype=tf.float32) # Local mins for h free
    y0_state = tf.Variable(tf.zeros((batch_size,target_size)),dtype=tf.float32) # Local mins for y
    h0_activity = tf.Variable(tf.zeros((batch_size,hidden_size)),dtype=tf.float32) # Local mins for h free
    y0_activity = tf.Variable(tf.zeros((batch_size,target_size)),dtype=tf.float32) # Local mins for y
    d = tf.Variable(tf.zeros((batch_size,target_size)),dtype=tf.float32) # Targets
    activity = tf.Variable(np.zeros(hidden_size),dtype=tf.float32) # Activity

    #########################
    # Set activation function
    #########################
    if activation_type == "hard":
        h = HardSigmoidLayer((batch_size,hidden_size))
        y = HardSigmoidLayer((batch_size,target_size))
    elif activation_type == "vslope":
        h = VarSlopeHSigLayer((batch_size,hidden_size),threshold=args.threshold,tau=args.tau,reset_val=args.reset_val,max_val=args.max_val)
        y = VarSlopeHSigLayer((batch_size,target_size),threshold=args.threshold,tau=args.tau,reset_val=args.reset_val,max_val=args.max_val)
    elif activation_type == "b_thresh":
        h = BinaryThresholdLayer((batch_size,hidden_size),threshold=args.threshold)
        y = BinaryThresholdLayer((batch_size,target_size),threshold=args.threshold)
    elif activation_type == "loihi":
        h = LoihiLIFLayer((batch_size,hidden_size),threshold=args.threshold,tau=args.tau)
        y = LoihiLIFLayer((batch_size,target_size),threshold=args.threshold,tau=args.tau)
    elif activation_type == "lif":
        h = NonSpikingLIFLayer((batch_size,hidden_size),threshold=args.threshold,tau=args.tau,reset_val=args.reset_val,max_val=args.max_val)
        y = NonSpikingLIFLayer((batch_size,target_size),threshold=args.threshold,tau=args.tau,reset_val=args.reset_val,max_val=args.max_val)

##################################################################################################################
##################################################################################################################

    #######
    # Train
    #######
    for epoch in range(n_epochs):
        if epoch != 0:
            learning_rate = [i * 0.75 for i in learning_rate]
        for batch in range(60000//batch_size): # Iterate over mini batches
            x.assign(images[batch*batch_size:(batch+1)*batch_size]) # Set input batch
            d.assign(targets[batch*batch_size:(batch+1)*batch_size]) # Set targets

            #########
            # Phase 1
            #########

            #Activations / firing rates
            h.updateActivity()
            y.updateActivity()
            for step in range(phase_1_n_steps):

                #Inputs
                if symmetric:
                    h_input = x@W_xh + y.activity()@tf.transpose(W_hy)
                else:
                    h_input = x@W_xh + y.activity()@W_yh
                y_input = h.activity()@W_hy

                if not no_bias:
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


            # Print to log
            if batch%50==0:

                # Get Predictions
                predictions = tf.math.argmax(y.activity(),axis=1)
                actual = tf.math.argmax(d,axis=1)
                accuracy = tf.reduce_mean(tf.cast(tf.math.equal(actual,predictions),tf.float32))
                hidden_activity = np.sum(h.activity().numpy()>1e-8)/batch_size

                run_log.write("\n\nRunning batch "+str(batch)+" / Epoch "+str(epoch+1)+"\n\n")
                run_log.write("Accuracy: "+str(100*accuracy.numpy())+"%\n\n")
                run_log.write("Average nodes active: "+str(hidden_activity)+" out of "+str(hidden_size)+"="+str(100*hidden_activity/hidden_size)+"%\n")

                # Average nodes active
                ana = tf.reduce_mean(tf.reduce_sum(tf.cast(h.activity()>1e-25,tf.float32),[1]))
                avg_nodes_active = ana.numpy()
                run_log.write("Active nodes at time " + str(batch) + " :" + str(avg_nodes_active) + "%\n")
                W_xh_norm = tf.sqrt(tf.reduce_mean(W_xh**2)).numpy()
                W_hy_norm = tf.sqrt(tf.reduce_mean(W_hy**2)).numpy()
                W_yh_norm = tf.sqrt(tf.reduce_mean(W_yh**2)).numpy()
                run_log.write("L2 norm of weights "+str(W_xh_norm)+", "+str(W_hy_norm)+", "+str(W_yh_norm))


            #########
            # Phase 2
            #########
            for step in range(phase_2_n_steps):

                #Save local mins
                h0_activity.assign(h.activity())
                y0_activity.assign(y.activity())
                h0_state.assign(h.getState())
                y0_state.assign(y.getState())

                # Inputs
                if symmetric:
                    h_input = x@W_xh + y.activity()@tf.transpose(W_hy)
                else:
                    h_input = x@W_xh + y.activity()@W_yh
                y_input = h.activity()@W_hy + beta*(d-y.activity())

                if not no_bias:
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

                ################
                # Update weights
                ################
                dh_state = h.getState()-h0_state
                dy_state = y.getState()-y0_state
                dh = h.activity()-h0_activity
                dy = y.activity()-y0_activity

                # Homeostatic options
                if norm == 'none':
                    dW_xh_homeostatic = 0
                    dW_hy_homeostatic = 0
                    dW_yh_homeostatic = 0
                elif norm == 'oja':
                    dW_xh_homeostatic = -tf.reduce_mean(h_input*dh,axis=(0,),keepdims=True)
                    dW_hy_homeostatic = -tf.reduce_mean(y_input*dy,axis=(0,),keepdims=True)
                    dW_yh_homeostatic = -tf.reduce_mean(h_input*dh,axis=(0,),keepdims=True)
                elif norm == 'linear': # Original with weight decay from "Biologically plausible models of homeostasis and STDP"
                    dW_xh_homeostatic = 1-tf.reduce_mean(h0_activity)/optimal
                    dW_hy_homeostatic = 1-tf.reduce_mean(y0_activity)/optimal
                    dW_yh_homeostatic = 1-tf.reduce_mean(h0_activity)/optimal
                elif norm == 'quadratic':
                    dW_xh_homeostatic = tf.reduce_mean(h0_activity*(1-h0_activity/optimal))
                    dW_hy_homeostatic = tf.reduce_mean(y0_activity*(1-y0_activity/optimal))
                    dW_yh_homeostatic = tf.reduce_mean(h0_activity*(1-h0_activity/optimal))

                # Compute weight updates
                if learning_rule == 0: # STDP rule
                    dW_xh = (beta0*tf.transpose(x)@h0_activity+beta1*tf.transpose(x)@dh)/batch_size
                    dW_hy = (beta0*tf.transpose(h0_activity)@y0_activity + beta1*(tf.transpose(h0_activity)@dy-tf.transpose(dh)@y0_activity))/batch_size
                    dW_yh = (beta0*tf.transpose(y0_activity)@h0_activity + beta1*(tf.transpose(y0_activity)@dh-tf.transpose(dy)@h0_activity))/batch_size
                elif learning_rule == 1: # Original
                    dW_xh = tf.transpose(x)@dh/batch_size
                    dW_hy = tf.transpose(h0_activity)@dy/batch_size
                    dW_yh = tf.transpose(y0_activity)@dh/batch_size
                elif learning_rule == 2: # Sister rule
                    dW_xh = tf.transpose(x)@dh/batch_size
                    dW_hy = -tf.transpose(dh)@y0_activity/batch_size
                    dW_yh = -tf.transpose(dy)@h0_activity/batch_size
                W_xh.assign_add(learning_rate[0]*dW_xh+decay_rate[0]*dW_xh_homeostatic*W_xh)
                if symmetric:
                    W_hy.assign_add(learning_rate[1]*(dW_hy+tf.transpose(dW_yh))/2)
                else:
                    W_hy.assign_add(learning_rate[1]*dW_hy+decay_rate[1]*dW_hy_homeostatic*W_hy)
                    W_yh.assign_add(learning_rate[1]*dW_yh+decay_rate[1]*dW_yh_homeostatic*W_yh)
                    

######################################################################################################################
######################################################################################################################

        ######
        # Test
        ######
        total_test_error = 0

        for batch in range(10000//batch_size): # Iterate over mini batches
            x.assign(test_images[batch*batch_size:(batch+1)*batch_size]) # Set input batch
            d.assign(test_targets[batch*batch_size:(batch+1)*batch_size]) # Set targets

            #Activations / firing rates
            h.updateActivity()
            y.updateActivity()
            for step in range(phase_1_n_steps):

                #Inputs
                if symmetric:
                    h.setInput( x@W_xh + y.activity()@tf.transpose(W_hy) )
                else:
                    h.setInput( x@W_xh+y.activity()@W_yh )
                y.setInput( h.activity()@W_hy )

                #Update potentials
                h.updateState(epsilon)
                y.updateState(epsilon)

                #Activations / firing rates
                h.updateActivity()
                y.updateActivity()

            # Get test accuracy
            predictions = tf.math.argmax(y.activity(),axis=1)
            actual = tf.math.argmax(d,axis=1)
            accuracy = tf.reduce_sum(tf.cast(tf.math.equal(actual,predictions),tf.float32))
            total_test_error += accuracy.numpy()

        total_test_error /= 10000

        run_log.write("\nTest accuracy "+str(100*total_test_error)+"%")
        print("Test accuracy ",100*total_test_error,"%")
        test_accs.append(100*total_test_error)

    np.save('W_xh.npy',W_xh)
    np.save('W_hy.npy',W_hy)
    np.save('W_yh.npy',W_yh)
    np.save('b_h.npy',b_h)
    np.save('b_y.npy',b_y)

    if args.symmetric:
        symmetry = 'symmetric'
    else:
        symmetry = 'asymmetric'

    fields = []
    for var in args_dict:
        fields.append(var)
        fields.append(args_dict[var])
    fields.append(test_accs)
    with open(outf, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    run_log.close()
