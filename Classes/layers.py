import tensorflow as tf
from abc import ABC, abstractmethod

######################
# Activation functions
######################

def binary_activation(x,threshold):
    cond = tf.less(x, tf.ones(tf.shape(x))*threshold)
    return tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

def threshold_relu(x,threshold):
    cond = tf.less(x, tf.ones(tf.shape(x))*threshold)
    return tf.where(cond, tf.zeros(tf.shape(x)), x)


class Layer(ABC):

    def __init__(self):
        self._activity = None
        self._input = None

    def setInput(self,dataIn):
        self._input = dataIn

    def getInput(self):
        return self._input

    def getState(self):
        return self._state

    def setState(self, val):
        self._state.assign(val)

    def setActivity(self,val):
        self._activity.assign(val)

    @abstractmethod
    def updateState(self, dataIn):
        pass

    @abstractmethod
    def updateActivity(self):
        pass

    @abstractmethod
    def activity(self):
        pass

    
class HardSigmoidLayer(Layer):

    def __init__(self, size: tuple, dtype=tf.float32):
        super().__init__()
        self.dtype = dtype
        self._state = tf.Variable(tf.zeros(size,dtype=self.dtype),dtype=self.dtype)

    def updateState(self,epsilon, mn = 0.0, mx = 1.0):
        self._state.assign( tf.clip_by_value( self._state*(1-epsilon) + epsilon*self._input, mn, mx))

    def updateActivity(self):
        self._activity = self._state

    def activity(self):
        return self._activity

    
class VarSlopeHSigLayer(Layer):

    def __init__(self, size: tuple, dtype=tf.float32, threshold=1e-6, tau=1e6, reset_val=0.0, max_val=1.0):
        super().__init__()
        self.dtype = dtype
        self._state = tf.Variable(tf.zeros(size,dtype=self.dtype),dtype=self.dtype)

        # reset_val <= threshold
        self.thresh = threshold
        self.tau = tau
        self.reset_val = reset_val
        self.max_val = max_val
        assert(self.reset_val<=self.thresh)

        # Compute parameters for linear funciton
        self.m = 1./(self.tau*(self.thresh - self.reset_val))
        self.h_shift = (self.thresh + self.reset_val)/2

    def updateState(self,epsilon):
        self._state.assign(tf.clip_by_value(self._state*(1-epsilon) + epsilon*self._input, 0, self.max_val/self.m+self.h_shift))

    def updateActivity(self):
        self._activity = self.m*threshold_relu(self._state-self.h_shift,self.thresh-self.h_shift)

    def activity(self):
        return self._activity

    
class BinaryThresholdLayer(Layer):

    def __init__(self, size: tuple, dtype=tf.float32, threshold=0.5):
        super().__init__()
        self.dtype = dtype
        self._state = tf.Variable(tf.zeros(size,dtype=self.dtype),dtype=self.dtype)
        self.thresh = threshold

    def updateState(self,epsilon):
        self._state.assign( self._state*(1-epsilon) + epsilon*self._input )

    def updateActivity(self):
        self._activity = binary_activation(self._state,self.thresh)

    def activity(self):
        return self._activity

    
class LoihiLIFLayer(Layer):

    def __init__(self, size: tuple, dtype=tf.float32, threshold=0.25, tau=2):
        super().__init__()
        self.dtype = dtype
        self.thresh = threshold
        self._state = tf.Variable(1e-20+self.thresh*tf.ones(size,dtype=self.dtype),dtype=self.dtype)
        self.tau = tau

    def updateState(self,epsilon):
        self._state.assign( tf.maximum( self._state*(1-epsilon) + epsilon*self._input, 1e-20))

    def updateActivity(self):
        self._activity = self._activity = 1./tf.math.ceil(-self.tau*tf.math.log(tf.math.maximum(1-self.thresh/self._state,1e-20)))

    def activity(self):
        return self._activity

    
class NonSpikingLIFLayer(Layer):

    def __init__(self, size: tuple, dtype=tf.float32, threshold=0.25, tau=2.0, reset_val=0.0, max_val=1.0):
        super().__init__()
        self.dtype = dtype
        self.thresh = threshold
        self.tau = tau
        self.reset_val = reset_val
        self.max_val = max_val
        self._state = tf.Variable(1e-30*tf.ones(size,dtype=self.dtype),dtype=self.dtype)
        assert(self.reset_val<=self.thresh)
    def updateState(self,epsilon):
        self._state.assign( tf.maximum( self._state*(1-epsilon) + epsilon*self._input, 1e-30))

    def updateActivity(self):
        self._activity = tf.minimum(1./(self.tau*(tf.math.log(tf.maximum(1-self.reset_val/self._state,1e-30))-tf.math.log(tf.maximum(1-self.thresh/self._state,1e-30)) )), self.max_val)

    def activity(self):
        return self._activity

    
class PoissonLayer(Layer):
    def __init__(self,size:tuple,dtype=tf.float32):
        super().__init__()
        self.dtype=dtype
        self._state = tf.Variable(tf.zeros(size,dtype=self.dtype),dtype=self.dtype)
        self._activity = tf.Variable(tf.zeros(size,dtype=self.dtype),dtype=self.dtype)
        self._input = tf.Variable(tf.zeros(size,dtype=self.dtype),dtype=self.dtype)
        self._trace = tf.Variable(tf.zeros(size,dtype=self.dtype),dtype=self.dtype)
        self._spike = tf.Variable(tf.zeros(size,dtype=self.dtype),dtype=self.dtype)
        self.size = size
    def updateState(self,epsilon, mn = 0.0, mx = 1.0):
        self._state.assign( tf.clip_by_value( self._state*(1-epsilon) + epsilon*self._input, mn, mx))
    def updateActivity(self):
        self._activity = self._state
    def updateTrace(self,b,a=1.0):
        decay = 1-b
        self._trace.assign (decay*(self._trace + a*self._spike))
    def resetTrace(self):
        self._trace.assign(self._trace*0.0)
    def updateSpike(self):
        self._spike.assign(tf.cast(tf.random.uniform(self.size,dtype=self.dtype)<self._activity,self.dtype))
    def trace(self):
        return self._trace
    def activity(self):
        return self._activity
    def spike(self):
        return self._spike