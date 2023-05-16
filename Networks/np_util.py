import numpy as np
from scipy.integrate import dblquad

def firingRates(t,example,duration=1.0):
    if example == 0:
        r=100*(1+np.sin(2+t*np.pi/3))
        s=100*(1-np.cos(-2+t*np.pi/5))
    elif example == 1:
        r=100*(1+np.sin(t*np.pi/2))
        s=100*(1+np.cos(1+t*np.pi/3))
    elif example == 2:
        r=30*(2-np.sin(2+t*np.pi/2))
        s=100*(1+np.cos(t*np.pi/3))
    elif example == 3:
        r=100*(1+np.sin(1+t*np.pi/3))
        s=100*(1+np.cos(1+t*np.pi/3))
    elif example == 4:
        r = 10 + 80*t
        s = 80 - 30*t
    elif example == 5: # 0-> 200, 200 -> 0
        r = 200*t/duration
        s = 200*(1-t/duration)
    elif example == 6:
        r = 200*t/duration
        s = 100+0*t
    elif example == 7:
        r = 200*t/duration
        s = 50 + 100*t/duration
    elif example == 8:
        r=100*(1+np.cos(1+t*np.pi/2))
        s=100*(1+np.cos(1+t*np.pi/2))
    elif example == 9:
        r = 200*t/duration
        s = 200*t/duration
    return r,s