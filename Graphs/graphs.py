import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.insert(0,"../../Classes")
sys.path.insert(0,"../Networks")
from np_util import firingRates

##################
 #STDP curve graph
 #################

# Define the x values
t_neg = np.linspace(-50, 0, 500)
t_pos = np.linspace(0, 50, 500)
t = np.linspace(-50,50,1001)

# Define the y values
tau = 20
y2 = 1*(t>0)*np.exp(-t/tau)-1*(t<0)*np.exp(t/tau)

# Create a new figure and set its size
plt.figure(figsize=(5, 5))

# Plot the function
plt.plot(t,y2,color='#004488')

# Set the x and y labels and title
plt.xlabel('$\Delta t$ (ms)',fontsize=18)
plt.ylabel('$\Delta W$',fontsize=18)
#plt.title('STDP curves')

# Set the x and y limits
plt.xlim(-50, 50)
plt.ylim(-1, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Turn on the grid
plt.grid()

# Show the plot
plt.savefig("STDP_curve.svg",format='svg',transparent=True,bbox_inches='tight')
plt.savefig("STDP_curve.png",bbox_inches='tight')


##############
# Firing rates
##############

# Define the x values
duration = 2.0
n_examples = 10
t = np.linspace(0, duration*1000, int(duration*1000)) # time in ms
r_list, s_list = [],[]

# Create a new figure and set its size
fig, axs = plt.subplots(5,2,figsize=(4.9,12.0))

handles = []  # initialize handles for the legend
labels = []   # initialize labels for the legend

for example in range(n_examples):
    r,s = firingRates(t/1000, example=example, duration=duration)
    r_list.append(r)
    s_list.append(s)

    # Plot the two functions and add the handles and labels
    pre_handle, = axs[example//2, example%2].plot(t, r, color='#004488')
    post_handle, = axs[example//2, example%2].plot(t, s, color='#bb5566')
    if example == 0:
        handles.extend([pre_handle, post_handle])
        labels.extend(['pre-synaptic firing rate', 'post-synaptic firing rate'])

    axs[example//2, example%2].set_xlabel('$t$ (ms)')
    axs[example//2, example%2].set_ylabel('Firing rate (Hz)')
    axs[example//2, example%2].set_title(chr(example+65))
    axs[example//2, example%2].grid(True)
    axs[example//2, example%2].set_xlim(0, duration*1000)
    axs[example//2, example%2].set_ylim(0, 200)

# Add the legend to the figure
fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.96), loc="upper center", fancybox=True, framealpha=0, edgecolor=(0, 0, 0, 1.))

for ax in fig.get_axes():
    ax.label_outer()

# Save the plot
plt.savefig("Firing_rates.svg",format='svg',transparent=True,bbox_inches='tight')
plt.savefig("Firing_rates.png",bbox_inches='tight')