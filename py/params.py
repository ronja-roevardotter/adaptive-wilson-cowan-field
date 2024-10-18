import numpy as np

import sys
import os

cdir = os.getcwd() #get current directory
os.chdir(cdir) #make sure you are in the current directory
sys.path.append(cdir) #append directory to enable import of functions from current directory files

import kernels as ks

"""
Note: any function here can be called speratly, but one has to use a dotdict-dictionary to call the functions and know, which parameters are used for computations.
"""
    
    

class dotdict(dict):
    """dot.notation access to dictionary attributes. dotdict = GEKLAUT von neurolib.utils.collections.py 
    (https://github.com/neurolib-dev/neurolib/blob/master/neurolib/utils/collections.py)
    Example:
    ```
    model.params['duration'] = 10 * 1000 # classic key-value dictionary
    model.params.duration = 10 * 10000 # easy access via dotdict
    ```
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # now pickleable!!!
    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)
  
    
def defaultParams():
    
    params = dotdict({})
    
    #membrane time constants [no units yet]
    params.tau_e = 10.0 #default 1 or 10.0? #2.5 
    params.tau_i = 15.0 #default 1.5 or 15.0? #3.75
    
    #coupling weights (determining the dominant type of cells, e.g. locally excitatory...)
    params.w_ee = 3.2 #excitatory to excitatory
    params.w_ei = 2.6 #inhibitory to excitatory
    params.w_ie = 3.3 #excitatory to inhibitory
    params.w_ii = 0.9 #inhibitory to inhibitory
    
    #threshold and gain factors for the sigmoidal activation functions 
    params.beta_e = 5 #excitatory gain
    params.beta_i = 5 #inhibitory gain
    params.mu_e = 0 #excitatory threshold
    params.mu_i = 0 #inhibitory threshold
    
    # # - - adaptation parameters - - # #
    
    #transfer function
    params.beta_a = 20 #default = 10 oder 20?
    params.mu_a = 0.4
    
    #strength and time constant - to turn adaptation off: set b=0
    params.b = 0.2 #0.5 - set it 0 until further notice (mostly to not accidentally run analysis with adaptation)
    params.tau_a = 320
    
    # # - - - - # #
    
    #Write seperate function for setting the parameters of the coupling function w(x), but set the function type:
    params.kernel = 'gaussian' #else choosable: exponential,...

    #for simplification:
    params.sigma_e = 1 #characterizes the spatial extent of the excitatory coupling
    params.sigma_i = 3 #characterizes the spatial extent of the inhibitory coupling
    
    
    #external input currents: oscillatory state for default params
    params.I_e = 0.0 
    params.I_i = 0.0

    #temporal 
    #choose number of integration time step in seconds
    #NOTE: such a high integration time step can only be used, if runge-kutta is used for integration.
    params.dt = 0.5 #[ms] : we assume ms per one unit-time => 0.5 is every half millisecond
    
    #choose time interval [start,end]=[0,T] [in ms]
    params.start_t = 0
    params.end_t = 60 * 1000 #50000 ms = 50s
    
    #spatial
    #choose number of pixels per axis
    params.n = 1600
    
    #choose spatial boundaries (intervals of spatial spread)
    params.length = 800 #length of spatial component [for delay computations assumed to be in mm]
    
    #delay parameters:  1. delayed -> to indicate if delays shalle be turned on (True) or not (False)
    #                   2. c -> velocity of activity (in whatever unit you want to define everything)
    params.delayed = False
    params.c = 10 #m/s velocity of activity in m/s
    
    #to enable the same initialisation, plant a seed:
    params.seed = True
    params.seed_amp = 0.1
    params.seed_func = 'prng'
    
    return params
    
def setTime(params):
    """
    Necessary params:
    :start_t: beginning of time
    :end_t: end of time
    :dt: integration time step
    
    returns: array time of time intervall
    """

    return np.arange(params.start_t, params.end_t + params.dt, params.dt)

def setSpace(params, shift=False):
    """
    Necessary params: 
    :length: length of ring circumference
    :n: number of pixels
    
    returns: x, dx
    :x: distance array from one pixel to all other pixels
    :dx: float of integration space step
    """
    
    x_inf, x_sup, cx, dx= -(params.length/2), +(params.length/2), 0, (params.length/params.n)
        
    num_steps = int(params.n/2)+1
    
    x1, dx = np.linspace(cx, x_sup, num_steps, retstep=True)

    if params.n%2==0:
        x2 = x1[1:-1]
        x2 = x2[::-1]
        x = np.concatenate((-x1,x2))
    else:
        dxx = dx/2
        x_inf += dxx
        x = np.arange(x_inf, x_sup, dx)
        
    if shift:
        x = np.fft.fftshift(x)
        return x, dx
    else:
        return x, dx
    
def ringValues(params):
    """
    Necessary parameters:
    :kernel: string, which spatial kernel shall be used (gaussian vs exponential)
    :sigma_e: float, excitatory spread
    :sigma_i: float, inhibitory spread
    :x: distance array
    :dx: float, integration space step
    
    returns: ke, ki
    :ke: array of by spatial kernel function weighted excitatory connectivity to others
    :ki: array of by spatial kernel function weighted inhibitory connectivity to others
    """
    
    kernel_func = getattr(ks, params.kernel)
    
    ke = kernel_func(params.sigma_e, params.x)
    ki = kernel_func(params.sigma_i, params.x)
    
    #normalize kernel w.r.t. integration space step dx
    alpha_e = np.sum(ke)*params.dx #normalisation factor for exc
    alpha_i = np.sum(ki)*params.dx #normalisation factor for inh
    
    #normalisation & consideration of integration step s.t. we don't have to consider that anymore later.
    ke *= params.dx #(params.dx/alpha_e) 
    ki *= params.dx #(params.dx/alpha_i)
    
    fourier_func = getattr(ks, 'f_' + params.kernel)
    
    ke_fft = (1/np.sqrt(2*np.pi)) * fourier_func(params.sigma_e, params.x)
    ki_fft = (1/np.sqrt(2*np.pi)) * fourier_func(params.sigma_i, params.x)
    
    return ke, ki, ke_fft, ki_fft

def setDelay(params):
    """ 
    Necessary params:
    :x: distance array
    :c: velocity of activity in [m/s]
    
    returns:
    :delay: array of integers that indicate the delay from pixel to pixel based on velocity c
    """
    
    d_syn = 0.3 #in [m/s] minimum synaptic delay - one can set delay=d_syn+(|x-y|/c)
    
    delay_x = abs(params.x) #assume distance in mm and velocity in m/s -> transform c into mm/s by multiplying with 1000
 #   params.c *= 1000 #ommitable, if transformation into ms happens as well - m/s=mm/ms !!
    delay = delay_x*(1/params.c) # in seconds
    
 #   delay *= (1/1000) #in milliseconds (1sec = 1000ms)
    delay = delay*(1/params.dt)# in time steps
    
    delay += 1
    delay = abs(delay.astype(int))
    
    return delay
    
    
def setParams(pDict):
    
    params = defaultParams()
    
    if pDict is not None:
        for k, val in zip(pDict.keys(), pDict.values()):
            params[k] = val
                
                
    #To make sure, that I do NOT reconnect nodes with themselves again, I need a constraint on my spatial spread.
    #Maximum spread can be the distance to the node furthest away (in a ring that would be max(l/2))
#    if params.sigma_e >= params.length/2:
#        temp = params.sigma_e
#        params.sigma_e = params.sigma_e/(params.length/2)
#        print('sigma_e=%.2f was initialised too large %.2f>=%.2f==length/2 -> reset to sigma_e/length=%.2f.' 
#              %(temp, temp, params.length/2, params.sigma_e))
#    elif params.sigma_i >= params.length/2:
#        temp = params.sigma_i
#        params.sigma_i = params.sigma_i/(params.length/2)
#        print('sigma_i=%.2f was initialised too large %.2f>=%.2f==length/2 -> reset to sigma_i/(length/2)=%.2f.' 
#              %(temp, temp, params.length/2, params.sigma_i))
        
    params.time = setTime(params)
    
    params.x, params.dx = setSpace(params, shift=False)
    
    params.ke, params.ki, params.ke_fft, params.ki_fft = ringValues(params)
    
    params.ke_fft = np.fft.fft(params.ke)
    params.ki_fft = np.fft.fft(params.ki)
    
    params.delay = setDelay(params)
                
    return params

    