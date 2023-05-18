# stdp_approx_backprop
Public repository for the paper "Using Realistic Spike Timing Dependent Plasticity (STDP) Rules to Approximate Backpropagation for Supervised Learning Tasks" by Ari J. Herman, Steven C. Nesbit, Edward Kim, and Garrett T. Kenyon 

Most results in the paper are from "main.sh" which calls "main.py". To test the spiking model, run "spiking.py", which should give around 84% test accuracy. For the rate-based STDP approximations, run "stdp.py" or "stdp.sh". The asymmetric rule is #2 and the skew-symmetric rule is #0. The other two rules correspond to other methods that can also be used to approximate STDP, but which we did not discuss in the paper.
