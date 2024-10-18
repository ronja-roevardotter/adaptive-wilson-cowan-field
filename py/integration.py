import sys
import os

cdir = os.getcwd() #get current directory
os.chdir(cdir) #make sure you are in the current directory
sys.path.append(cdir) #append directory to enable import of functions from current directory files

import numpy as np

from params import setParams

def run(params=None, fp=np.array([0.0, 0.01]), wavenumber=1, itype='rungekutta'):
    """
            
    This is the model implementation for the continuum Wilson-Cowan model, called mtype = 'activity'
    or else the continuum Amari model, called mtype = 'voltage'.
    System type: Integro-Differential equations with temporal and spatial component => dim(system)=2d
    IDEs for excitatory & inhibitory populations (coupled)
    Spatial dimension: dim(x)=1d, arranged on a ring 
    -> all-to-all same connectivity, determined by spread, omitting boundary conditions
            
        
    List of possible keys+values in params-dict:

    parameters:
    :w_ee: excitatory to excitatory coupling, float
    :w_ei: inhibitory to excitatory coupling, float
    :w_ie: excitatory to inhibitory coupling, float
    :w_ii: inhibitory to inhibitory coupling, float
        
    :tau_e: excitatory membrane time constant, float
    :tau_i: inhibitory membrane time constant, float
        
    :beta_e: excitatory gain (in sigmoidal transfer function), float
    :beta_i: inhibitory gain (in sigmoidal transfer function), float
    :mu_e: excitatory threshold (in sigmoidal transfer function), float
    :mu_i: inhibitory threshold (in sigmoidal transfer function), float
        
    :I_e: external input current to excitatory population, float
    :I_i: external input current to inhibitory population, float
        
    :kernel: what function used to determine spatial kernel, string, options are 'gaussian' or 'exponential'
    :sigma_e: characterizes the spatial extent of the excitatory to [...] connectivity, float
    :sigma_i: characterizes the spatial extent of the inhibitory to [...] connectivity, float

    mechanism parameters:
    :beta_a: mechanism gain in sigmoidal transfer function, float (>0 if adaptation, <0 if h-current)
    :mu_a: mechanism threshold in sigmoidal transfer function, float
    :tau_i: mechanism time constant, float
    :b: mechanism strength, float  (>0 if adaptation, <0 if h-current)
        
    temporal components:
    :dt: integration time step, float -> observe ms, therefore, setting e.g. dt=0.1, means we look at every 10th ms.
    :start_t: start of time intervall, integer
    :end_t: end of time intervall, integer
        
    spatial components:
    :n: number of pixels/positions on ring, integer
    :length: length of total circumference of ring, float 
             (remark: max. distance from pixel to furthest away can bi maximally length/2)
    :c: velocity of activity in [m/s], float -> transformed into mm/s in py.params.setParams()
        
    created by given params:
    :x: array of distances from one pixel to all other pixels (same distances to left and right, omit boundary effects), array
    :dx: spatial integration step, determined by length and n, float
    :ke: kernel values (i.e. connectivity strengths) from excitatory population of a pixel to all other pixels, 
         determined by x, array
    :ki: kernel values (i.e. connectivity strengths) from excitatory population of a pixel to all other pixels, 
         determined by x, array
    :ke_fft: Fast Fourier Transform of ke by np.fft.fft, array
    :ki_fft: Fast Fourier Transform of ki by np.fft.fft, array 
    :time: array of time intervall, array
    :delay: temporal delay from one pixel to another, determined by x,c and dt, array    
    """

    params = setParams(params)
    
    return runIntegration(params, fp=fp, wavenumber=wavenumber, itype=itype)

def kernSeed(array, kernel, seed_amp):
    array += kernel*seed_amp
    return array

#pseud-random-number-generator for reproducible initialisations
def prngSeed(shape, fp, std, seed=42):

    # Set the seed for reproducibility
    np.random.seed(seed)
    
    # Generate random numbers with a normal distribution around 0
    noise = np.random.normal(loc=0, scale=std, size=shape)
    # Add the fixed value to the noise
    jittered_array = fp + noise

    return jittered_array 

def sinusSeed(array, length, wavenumber, seed_amp, x):
    wavelength = 2*np.pi / wavenumber
    m = length/wavelength
    wave_in_pi = (2*np.pi*m) / length
    x = np.roll(np.fft.fftshift(x), -1)
    phi = np.pi
    
    #for k in range(1,21): 
    #    array += seed_amp * (1/k) * wave_in_pi *  np.sin(k * wave_in_pi * x)
    
    array += seed_amp * (np.cos(wave_in_pi * x) + 0.2 * np.sin(2 * wave_in_pi * x + phi))
    
    #array += seed_amp * (wave_in_pi *  np.sin(wave_in_pi * x) + 
    #              0.5 * wave_in_pi * np.sin(2 * wave_in_pi * x ) + #+ phi) + 
    #              0.25 * wave_in_pi * np.sin(3 * wave_in_pi * x ) + #+ phi) + 
    #              0.125 * wave_in_pi * np.sin(4 * wave_in_pi * x ) + #+ phi) + 
    #              0.0625 * wave_in_pi * np.sin(5 * wave_in_pi * x ) +
    #              0.00315 * wave_in_pi * np.sin(5 * wave_in_pi * x ))

    #fit as many sinus waves into the ring as the length allows w.r.t. the max spatial wavenumber for which the turing state lost stability
    #not needed anymore: xes = np.linspace(0,2*np.pi*int(length/wavelength), len(array)) 
    #array += seed_amp * (np.cos(wave_in_pi * x) + 0.2 * np.sin(2 * wave_in_pi * x + phi)) #introducing slight asymmetry to ensure the waves travel into the same direction
    return array

def runIntegration(params, fp=np.array([0.0, 0.01]), wavenumber=1, itype='fourier'):
    
    """
    Before we can run the integration-loop, we have to set the parameters and call the integration by them 
    s.t. nothing's accidentally overwritten.
    """
    
    #membrane time constants [no units yet]
    tau_e = params.tau_e
    tau_i = params.tau_i
    
    #coupling weights (determining the dominant type of cells, e.g. locally excitatory...)
    w_ee = params.w_ee
    w_ei = params.w_ei
    w_ie = params.w_ie
    w_ii = params.w_ii
    
    #threshold and gain factors for the sigmoidal activation functions 
    beta_e = params.beta_e
    beta_i = params.beta_i
    mu_e = params.mu_e
    mu_i = params.mu_i
    
    # # - - mechanism parameters - - # #
    
    #transfer function
    beta_a = params.beta_a
    mu_a = params.mu_a
    
    #strength and time constant
    b = params.b
    tau_a = params.tau_a
    
    # # - - - - # #
    
    #external input currents: oscillatory state for default params
    I_e = params.I_e
    I_i = params.I_i

    #temporal 
    dt = params.dt
    
    #spatial
    n = params.n
    length = params.length
    c = params.c
    
    x = params.x
    dx = params.dx
    time = params.time
    
    ke = params.ke
    ki = params.ki
    
    ke_fft = params.ke_fft
    ki_fft = params.ki_fft
    
    delayed = params.delayed
    delay = params.delay
    
    seed =  params.seed
    seed_amp =  params.seed_amp
    seed_func = params.seed_func
    
    comparison = fp==[0.0,0.01]
    
    if all(comparison):
        init_exc = fp
        init_inh = fp
        init_adaps = fp
    else:
        a_fp = 1/(1+np.exp(-beta_a*(fp[0]-mu_a)))
        init_exc = [fp[0]-1e-10, fp[0]+1e-10]
        init_inh = [fp[1]-1e-10, fp[1]+1e-10]
        init_adaps = [a_fp-1e-10, a_fp+1e-10]
        
    if seed and not all(comparison): 
        ue_init = np.ones((len(time),n))*fp[0] #leads to [rows, columns] = [time, pixels (space)]
        ui_init = np.ones((len(time),n))*fp[1]
        adaps_init = np.ones((len(time),n))*a_fp

        #usually now I distinguish different seed functions but I only have one yet, so ...
        if seed_func == 'kern':
            ue_init[0] = kernSeed(ue_init[0], ke, seed_amp)
            ui_init[0] = kernSeed(ui_init[0], ki, seed_amp)
            adaps_init[0] = kernSeed(adaps_init[0], ke, seed_amp)
        elif seed_func == 'sinus':
            ue_init[0] = sinusSeed(ue_init[0], length, wavenumber, seed_amp, x)
            ui_init[0] = sinusSeed(ui_init[0], length, wavenumber, seed_amp, x)
            adaps_init[0] = sinusSeed(adaps_init[0], length, wavenumber, seed_amp, x)
        else:
            ue_init[0] = prngSeed(n, fp[0], seed_amp, 42)
            ui_init[0] = prngSeed(n, fp[1], seed_amp, 42)
            adaps_init[0] = prngSeed(n, a_fp, seed_amp, 42)
    else:
        #the initialisation I have to make to start the integration
        ue_init = np.zeros((len(time),n)) #leads to [rows, columns] = [time, pixels (space)]
        ui_init = np.zeros((len(time),n))
        adaps_init = np.zeros((len(time),n))
        ue_init[0]=np.random.uniform(init_exc[0], init_exc[1], n)
        ui_init[0]=np.random.uniform(init_inh[0], init_inh[1], n)
        adaps_init[0]=np.random.uniform(init_adaps[0], init_adaps[1], n)
    
    
    if itype=='convolution':
        integrate = globals()['integrate_approxi']
    elif itype=='rungekutta':
        integrate = globals()['integrate_runge_kutta']
    else: 
        integrate = globals()['integrate_fourier']
    
    ue, ui, adaps =  integrate(tau_e, tau_i,
                        w_ee, w_ei, w_ie, w_ii,
                        beta_e, beta_i, mu_e, mu_i,
                        I_e, I_i,
                        beta_a, mu_a, b, tau_a, adaps_init,
                        dt, time, delayed, delay, 
                        n, length, c, x, dx, 
                        ke, ki, ke_fft, ki_fft,
                        ue_init, ui_init)
    
    
    return ue, ui, adaps

# # # - - - Integration by Fourier Transform, Product, Inverse Fourier Transform of convolution; with and without adaptation - - - # # #
# # # - - - set params.b==0 to deactivate adaptation - - - # # #

def integrate_fourier(tau_e, tau_i,
             w_ee, w_ei, w_ie, w_ii,
             beta_e, beta_i, mu_e, mu_i,
             I_e, I_i,
             beta_a, mu_a, b, tau_a, adaps, 
             dt, time, delayed, delay, 
             n, length, c, x, dx, 
             ke, ki, ke_fft, ki_fft,
             ue, ui):

    def Fe(x):
        return 1/(1+np.exp(-beta_e*(x-mu_e)))
    
    def Fi(x):
        return 1/(1+np.exp(-beta_i*(x-mu_i)))
    
    def Fa(x):
        return 1/(1+np.exp(-beta_a*(x-mu_a)))
    
    if delayed:
        d_max = max(delay)
    else:
        d_max = len(time)
    
    ke_fft = np.fft.fft(ke)
    ki_fft = np.fft.fft(ki)
    
    for t in range(1,int(len(time))): 
        
        #indeces for delays ('how far back in time - index') - makes the delayed time steps easier to call
        indeces = np.array([t*np.ones(n)-delay]).astype(int)
        #per node, i.e. node 0 has delay 1 -> call ue[t-1,0], node 1 has delay 2 -> call ue[t-2, 1] etc.
        node_indeces = np.linspace(0,n-1,n).astype(int)

        #this distinction is only interesting when delays are turned on
        if t<=d_max+1: #len(time): 
            ve = np.fft.fft(ue[t-1])
            vi = np.fft.fft(ui[t-1])
        else:
            temp_e = ue[indeces,node_indeces]
            temp_i = ui[indeces,node_indeces]
            ve = np.fft.fft(temp_e)
            vi = np.fft.fft(temp_i)
                
        Le = ke_fft * ve
        Li = ki_fft * vi
        
        conv_e = np.fft.ifft(Le).real
        conv_i = np.fft.ifft(Li).real
        
        
        #determine the RHS before integrating over it w.r.t. time t
        rhs_adaps = ((1/tau_a)*(-adaps[t-1] + Fa(ue[t-1])))
        rhs_e = ((1/tau_e)*(-ue[t-1] + Fe(w_ee*conv_e - w_ei*conv_i - b*adaps[t-1] + I_e)))
        rhs_i = ((1/tau_i)*(-ui[t-1] + Fi(w_ie*conv_e - w_ii*conv_i + I_i)))
        
        #integrate with euler integration
        ue[t] = ue[t-1] + (dt * rhs_e)
        ui[t] = ui[t-1] + (dt * rhs_i)
        adaps[t] = adaps[t-1] + (dt * rhs_adaps)
        
    
    return ue, ui, adaps

def integrate_runge_kutta(tau_e, tau_i,
                      w_ee, w_ei, w_ie, w_ii,
                      beta_e, beta_i, mu_e, mu_i,
                      I_e, I_i,
                      beta_a, mu_a, b, tau_a, adaps,
                      dt, time, delayed, delay,
                      n, length, c, x, dx,
                      ke, ki, ke_fft, ki_fft,
                      ue, ui):
    
    """"
    NOTE: The Runge-Kutta Implementation - as of right now - does NOT work with delays!!!
    """

    def Fe(x):
        return 1 / (1 + np.exp(-beta_e * (x-mu_e)))

    def Fi(x):
        return 1 / (1 + np.exp(-beta_i * (x-mu_i)))

    def Fa(x):
        return 1 / (1 + np.exp(-beta_a * (x-mu_a)))

    if delayed:
        d_max = len(time) #max(delay) - because it doesn't work with delays => turn them of.
    else:
        d_max = len(time)

    ke_fft = np.fft.fft(ke)
    ki_fft = np.fft.fft(ki)

    for t in range(1, int(len(time))):

        #indeces for delays ('how far back in time - index') - makes the delayed time steps easier to call
      #  indeces = np.array([t * np.ones(n)-delay]).astype(int)
        #per node, i.e. node 0 has delay 1 -> call ue[t-1,0], node 1 has delay 2 -> call ue[t-2, 1] etc.
        node_indeces = np.linspace(0, n-1, n).astype(int)

        if t <= d_max + 1:  # len(time):
            ve = np.fft.fft(ue[t-1])
            vi = np.fft.fft(ui[t-1])
        else:
            temp_e = ue[indeces, node_indeces]
            temp_i = ui[indeces, node_indeces]
            ve = np.fft.fft(temp_e)
            vi = np.fft.fft(temp_i)

        Le = ke_fft * ve
        Li = ki_fft * vi

        conv_e = np.fft.ifft(Le).real
        conv_i = np.fft.ifft(Li).real

        def compute_rhs(ue, ui, adaps):
            rhs_adaps = (1 / tau_a) * (-adaps + Fa(ue))
            rhs_e = (1 / tau_e) * (-ue + Fe(w_ee * conv_e - w_ei * conv_i - b * adaps + I_e))
            rhs_i = (1 / tau_i) * (-ui + Fi(w_ie * conv_e - w_ii * conv_i + I_i))
            return rhs_e, rhs_i, rhs_adaps

        #collect 4 order terms of rhs for Runge-Kutta
        k1_e, k1_i, k1_adaps = compute_rhs(ue[t-1], ui[t-1], adaps[t-1])
        k2_e, k2_i, k2_adaps = compute_rhs(ue[t-1] + 0.5 * dt * k1_e, ui[t-1] + 0.5 * dt * k1_i, adaps[t-1] + 0.5 * dt * k1_adaps)
        k3_e, k3_i, k3_adaps = compute_rhs(ue[t-1] + 0.5 * dt * k2_e, ui[t-1] + 0.5 * dt * k2_i, adaps[t-1] + 0.5 * dt * k2_adaps)
        k4_e, k4_i, k4_adaps = compute_rhs(ue[t-1] + dt * k3_e, ui[t-1] + dt * k3_i, adaps[t-1] + dt * k3_adaps)

        ue[t] = ue[t-1] + (dt / 6) * (k1_e + 2 * k2_e + 2 * k3_e + k4_e)
        ui[t] = ui[t-1] + (dt / 6) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        adaps[t] = adaps[t-1] + (dt / 6) * (k1_adaps + 2 * k2_adaps + 2 * k3_adaps + k4_adaps)

    return ue, ui, adaps


# # # - - - Integration with integral approximation of convolution - - - # # #
# # # - - -   could be used to try out e.g. heterogeneous kernel   - - - # # #
# # # - - -                        very slow                       - - - # # #

def integrate_approxi(tau_e, tau_i,
                 w_ee, w_ei, w_ie, w_ii,
                 beta_e, beta_i, mu_e, mu_i,
                 I_e, I_i,
                 beta_a, mu_a, b, tau_a, adaps, 
                 dt, time, delayed, delay, 
                 n, length, c, x, dx, 
                 ke, ki, ke_fft, ki_fft,
                 ue, ui):
    
    def Fe(x):
        return 1/(1+np.exp(-beta_e*(x-mu_e)))
    
    def Fi(x):
        return 1/(1+np.exp(-beta_i*(x-mu_i)))
    
    def Fa(x):
        return 1/(1+np.exp(-beta_a*(x-mu_a)))
    
    d_max = max(delay)
    indices = np.linspace(0,n-1, n).astype(int)
    
    ke_mtx = np.zeros((n,n))
    ki_mtx = np.zeros((n,n))
    delay_mtx = np.zeros((n,n)).astype(int)

    if delayed:
        d_max = max(delay)
    else:
        d_max = len(time)
    
    for j in range(n):
        ke_mtx[j] = np.roll(ke, j)
        ki_mtx[j] = np.roll(ki, j)
        delay_mtx[j] = np.roll(delay, j).astype(int)
    
    for t in range(1,int(len(time))): 
        
        L_e=np.zeros(n)
        L_i=np.zeros(n)
        
        if t<=d_max+1: #len(time):
            for j in range(n):
                L_e[j] = (ke_mtx[j] @ ue[t-1])
                L_i[j] = (ki_mtx[j] @ ui[t-1])
        else:
            for j in range(n):
                temp_e = ue[t-delay_mtx[j], indices]
                temp_i = ui[t-delay_mtx[j], indices]
                L_e[j] = (ke_mtx[j] @ temp_e)
                L_i[j] = (ki_mtx[j] @ temp_i)
            
        conv_e = L_e
        conv_i = L_i
        
        
        #determine the RHS before integrating over it w.r.t. time t
        rhs_adaps = ((1/tau_a)*(-adaps[t-1] + Fa(ue[t-1])))
        rhs_e = ((1/tau_e)*(-ue[t-1] + Fe(w_ee*conv_e - w_ei*conv_i - b*adaps[t-1] + I_e)))
        rhs_i = ((1/tau_i)*(-ui[t-1] + Fi(w_ie*conv_e - w_ii*conv_i + I_i)))
        
        #integrate with euler integration
        ue[t] = ue[t-1] + (dt * rhs_e)
        ui[t] = ui[t-1] + (dt * rhs_i)
        adaps[t] = adaps[t-1] + (dt * rhs_adaps)


    return ue, ui, adaps