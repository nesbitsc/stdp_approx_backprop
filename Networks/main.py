import numpy as np
import time
from matplotlib import pyplot as plt
import argparse
import csv
import sys
sys.path.insert(0,"../../Classes")
sys.path.insert(0,"../")
from np_util import firingRates

parser = argparse.ArgumentParser(description='Test firing rate approximations to all pairs STDP')
parser.add_argument('--a1', type=float,default=1.0, help='positive STDP weight')
parser.add_argument('--a2', type=float,default=-1.0, help='negative STDP weight')
parser.add_argument('--tau', type=float,default=0.02, help='set STDP time constant')
parser.add_argument('--t_step', type=float,default=1e-4, help='set time step size')
parser.add_argument('--example', type=int, default=0, help='activities curves for the two cells')
parser.add_argument('--n_itrs', type=int, default=100, help='set number of samples for STDP update')
parser.add_argument('--n_steps', type=int, default=30000, help='set number of samples for STDP update')
parser.add_argument('--duration', type=float,default=2.0, help='set t')
parser.add_argument('--trial_average', action='store_true',help='compute trial average (slow)')
args = parser.parse_args()

outf = 'output.csv'

n_steps=args.n_steps
duration = args.duration
t_step = duration/n_steps

eps = 10*args.tau # Window is [-eps,eps]

def stdp(t,tau=args.tau,a1=args.a1,a2=args.a2):
    return a1*(t>0)*(t<eps)*np.exp(-t/tau)+a2*(t<0)*(t>-eps)*np.exp(t/tau)# Positive 1,3,0,2

beta_0 = (args.a1+args.a2)*args.tau
beta_minus = -args.a2*args.tau**2
beta_plus = args.a1*args.tau**2

print("beta_0 = ",beta_0)
print("beta_minus = ",beta_minus)
print("beta_plus = ",beta_plus)

t = np.linspace(0.0,duration,n_steps)

def w(pre_rates,post_rates):
    pre_spikes = np.random.random(n_steps)<pre_rates*t_step
    post_spikes = np.random.random(n_steps)<post_rates*t_step

    pre_times = np.reshape(np.where(pre_spikes),(-1,1))
    post_times = np.reshape(np.where(post_spikes),(1,-1))

    diffs = (post_times-pre_times)*t_step
    dw = np.sum(stdp(diffs))
    del pre_spikes
    del post_spikes
    del pre_times
    del post_times
    return dw

def est_opt(pre_rates,post_rates):
    xx, yy = np.meshgrid(t,t)
    diffs = np.clip(xx-yy,-eps,eps)
    return pre_rates.T@stdp(diffs)@post_rates*t_step**2

def est(pre_rates,post_rates):
    pre_rate_derivative = np.diff(pre_rates,prepend=pre_rates[0])/t_step
    post_rate_derivative = np.diff(post_rates,prepend=post_rates[0])/t_step

    a = t_step*np.sum(pre_rates*post_rates)
    b = t_step*np.sum(pre_rates*post_rate_derivative)
    c = t_step*np.sum(post_rates*pre_rate_derivative)
    return a*beta_0+(b-c)*(beta_minus+beta_plus)/2, a*beta_0+b*beta_minus-c*beta_plus, a*beta_0+b*(beta_minus+beta_plus), a*beta_0-c*(beta_minus+beta_plus)

r,s = firingRates(t,example=args.example,duration=duration)

tests = []

tic = time.time()

fr_est0,fr_est1, fr_est2, fr_est3 = est(r,s)
fr_est_opt = est_opt(r,s)
toc = time.time()

print("Total time: ",toc-tic,"s")


if args.trial_average:
    for i in range(args.n_itrs):
        tests.append(w(r,s))
    trial_average = np.mean(tests)
    print("Trial average = ",trial_average)
    print("Best firing rate estimate = ",fr_est_opt)
    print("Error from spike correlation = ",np.abs( (trial_average - fr_est_opt) / trial_average ))
    print("Firing rate estimate 0 = ",fr_est0," / Error = ",np.abs( (trial_average-fr_est0) / trial_average ))
    print("Firing rate estimate 1 = ",fr_est1," / Error = ",np.abs( (trial_average-fr_est1) / trial_average ))
    print("Firing rate estimate 2 = ",fr_est2," / Error = ",np.abs( (trial_average-fr_est2) / trial_average ))
    print("Firing rate estimate 3 = ",fr_est3," / Error = ",np.abs( (trial_average-fr_est3) / trial_average ))
else:
    # Compare with optimal estimator
    print("Best firing rate estimate = ",fr_est_opt)
    print("Firing rate estimate 0 = ",fr_est0," / Error = ",np.abs( (fr_est_opt-fr_est0) / fr_est_opt ))
    print("Firing rate estimate 1 = ",fr_est1," / Error = ",np.abs( (fr_est_opt-fr_est1) / fr_est_opt ))
    print("Firing rate estimate 2 = ",fr_est2," / Error = ",np.abs( (fr_est_opt-fr_est2) / fr_est_opt ))
    print("Firing rate estimate 3 = ",fr_est3," / Error = ",np.abs( (fr_est_opt-fr_est3) / fr_est_opt ))

dictionary = vars(args)
fields = []
for var in dictionary:
    fields.append(dictionary[var])
if args.trial_average:
    fields.append(trial_average)
fields.append(fr_est_opt)
fields.append(fr_est0)
fields.append(fr_est1)
fields.append(fr_est2)
fields.append(fr_est3)

with open(outf, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)

del r,s,t
