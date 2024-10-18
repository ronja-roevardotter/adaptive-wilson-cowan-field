import numpy as np
import scipy.signal as signal 
from scipy.signal import find_peaks
from scipy.signal import hilbert

#This py-file includes functions to process arrays, data, compute something. Some of them are rather specific, some are rather general, all of them are used at least once for analysis reasons.


def getSwitchArray(array):
    """ returns the array at which positions the x-axis (have a zero) is crossed
    help for implementation found on 'stackoverflow.com'
    INPUT:
    :array: 1d array of numerical values
    
    OUPUT:
    :sign_switch_array: array with 1 at position of switch
    e.g. array=[2,3,1,-1,-2,-1,1,2] -> sign_switch_array = [0, 0, 0, 1, 0, 0, 1, 0]
    """
    
    
    signs = np.sign(array)
    sign_switch_array = ((np.roll(signs, 1) - signs) != 0).astype(int)
    
    return sign_switch_array

def getSwitchIndex(array):
    """ returns the indeces at which the positions the x-axis (have a zero) is crossed
    help for implementation found on 'geeksforgeeks.org'
    INPUT:
    :array: 1d array of numerical values
    
    OUPUT:
    :idx: array with indeces of switch
    e.g. array=[2,3,1,-1,-2,-1,1,2] -> idx = [3,6]
    """
    
    sign_switch_array = getSwitchArray(array)
    idx = np.where(sign_switch_array == 1)[0]

    if len(idx) > 0 and idx[0] == 0:
        #Find the first index greater than 0
        first_non_zero_index = next((i for i, x in enumerate(idx) if x > 0), None)
        if first_non_zero_index is not None:
            idx = idx[first_non_zero_index:]
    
    return idx

def getCommonElement(array1, array2):
    """"
    This function finds if array1 and array2 have an entry in common.
    INPUT:
    :array1: numpy array of elements
    :array2: numpy array of elements

    OUTPUT: 
    :bool: True if yes, False if no.
    """

    array1_set = set(array1)
    array2_set = set(array2)
    if (array1_set & array2_set):
        return True
    else:
        return False
    
    

def getPSD(array, fs, maxfreq=300, nperseg = 1):
    """returns the Powerspectrum (density: [V**2/Hz]) with the possibility to cut off the PSD at a maximum frequency
    
    INPUT:
    :array: time series (type: numpy-array)
    :fs: sampling frequency of array (e.g. variables['delta_t'] for array over time, variables['n'] for array over space)
    :maxfreq: maximum frequency to observe
    
    OUTPUT:
    :freqs: array of all frequencies that are looked at for the PDS
    :PSD_den: Power Spectrum Density (i.e. returns the power per frequency over the freqs array)
    """
    
    freqs, Pxx_den = signal.welch(array, fs, window='hann', nperseg=int(abs(nperseg*fs)))
    
    if maxfreq==None:
        maxfreq = max(freqs)
    
    freqs = freqs[freqs < maxfreq]
    Pxx_den = Pxx_den[0 : len(freqs)]
    
    return freqs, Pxx_den



def getAvgPSD(arrays, fs, maxfreq=None, nperseg=10):
    """Returns the average Power Spectrum Density for a mxn-dimensional array.
    
    INPUT:
    :arrays: mxn-dimensional array (m rows, n columns)
    :fs: sampling frequency
    :maxfreq: maximum frequency
    
    OUTPUT:
    :freqs: array of all frequencies that are looked at for the avg-PSD
    :avg_Pxx_den: average PSD (averages over rows)
    
    e.g. I give the array exc with dimension 37x4000
    then the PSD will be computed for frequencies over time and averaged over rows, which are the nodes (space)
    => returned avg_Pxx_den indicates the power per frequencies over time averaged over each node
    i.e. in that case, if the avg_Pxx_den is close to zero everywhere, I do not have a change in activity over time => temporally homogeneous
    
    """
    
    freqs, temp_Pxx_den = getPSD(arrays[0], fs, maxfreq, nperseg)
    
    all_Pxx_den = np.zeros((int(arrays.shape[0]),(len(temp_Pxx_den))))
    
    for idx, array in enumerate(arrays[1:]):
        f, Pxx_den = getPSD(array, fs, maxfreq, nperseg)
        all_Pxx_den[idx+1] = Pxx_den
        
    avg_Pxx_den = np.mean(all_Pxx_den, axis=0)
        
    
    return freqs, avg_Pxx_den


def getPosition(pixel_nmb, params):
    """
    This function returns the position on the ONE-dimensional line of a certain pixel.
    Input:
    :pixel_nmb: number of the pixel, integer
    :params: only necessary - params.x
    
    Output:
    :position: position x on line.
    """
    
    position = params.x[pixel_nmb]
    
    return position

# e
def normalize(arr, t_min, t_max):
    """"
    Explicit function to normalize array (for visualisation reasons - very helpful!)
    INPUT:
    :arr: one-dimensional numpy array of numerical values
    :t_min, t_max: upper and lower bound of range into which arr shall be normalized

    OUTPUT:
    :norm_arr: normalized numpy array with min(norm_arr)=t_min, max(norm_arr)=t_max
    """
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

# # # # # # # - - - - - - - - THE BELOW FUNCTIONS ARE USED TO COMPUTE PHASE lATENCIES IN A 1-DIMENSIONAL MODEL - - - - - - - - # # # # # # # 
# # # - - - As of right now, the functions are used for spatiotemporal patterns of periodic traveling waves - - - # # #
# # # - - - The pipeline, how it is done, is explained in a separate file (visu_phase_latency_pipeline.py) - - - # # #

def rotation_in_latency(array):
    """This function determines, based on the phase latency, into which direction the traveling waves rotate.
    INPUT:
    :array: numpy array, 1-dimensional, consists of the phase latency 
            ('how many time steps it takes node to cross the threshold 2\pi') per node (i.e. dim(array)=params.n)
            
    OUTPUT:
    :rotation: -1 (clockwise), or +1 (counterclockwise)
    """
    
    max_arg = np.argmax(array)
    argmax_before = max_arg-1
    argmax_after = max_arg+1
    if argmax_after == len(array):
        argmax_after=0
    diff_to_left = np.abs(array[max_arg]-array[int(argmax_before)])
    diff_to_right = np.abs(array[max_arg]-array[int(argmax_after)])
    
    if diff_to_left>diff_to_right:
        rotation = +1 #"counterclockwise"
    else:
        rotation = -1 #"clockwise"

    return rotation

def count_nodes_for_descent(array):
    """This function determines the amount of nodes that are necessary for one full phase transition from max to min
    i.e. how many nodes are necessary for one full phase.

    INPUT: 
    :array: numpy array, 1-dimensional, consists of the phase latency 
            ('how many time steps it takes node to cross the threshold 2\pi') per node (i.e. dim(array)=params.n)
    
    OUTPUT:
    :count: amount of nodes that are necessary until the next full phase transition starts
    """

    peaks, _ = find_peaks(array)
    values = array[peaks]

    if len(values) == 0:
        return 0, 0, 0

    #get the index for the node that requires the longest time to complete one temporal oscillation
    max_peak = peaks[np.argmax(values)]
    max_peak_ind = np.argmax(values)
    
    if max_peak_ind+1>=len(values):
        next_ind = 0
    else:
        next_ind = int(max_peak_ind+1)
    
    #get the index of node that starts a new temporal oscillation cycle
    next_max_peak = np.max([peaks[int(max_peak_ind-1)], peaks[next_ind]])

    #get the amount of nodes necessary to travel that one full spatial oscillation
    amount_of_nodes = np.abs(max_peak - next_max_peak) + 1

    #identify in which direction (clockwise vs. counter-clockwise) the activity is traveling
    if next_max_peak < max_peak:
        peaks = peaks[::-1]
    else:
        peaks=peaks

    #get differences between consecutive node indeces where temporal oscillations start their cycle
    peak_distances = [np.abs(j - i) for i, j in zip(peaks[:-1], peaks[1:])]

    #get the average amount of nodes necessary to travel full spatial oscillation-periods
    avg_amount_of_nodes = np.mean(peak_distances)

    #double the amount of nodes to account for a full spatial oscillation and not only half
    amount_of_nodes *= 2
    avg_amount_of_nodes *=2

    return amount_of_nodes, avg_amount_of_nodes, peaks

def count_avg_nodes_for_descent(array, rotation):
    """This function determines the amount of nodes that are necessary for one full phase transition from each max to each min
    i.e. how many nodes are necessary for one full phase.
    
    INPUT: 
    :array: numpy array, 1-dimensional, consists of the phase latency 
            ('how many time steps it takes node to cross the threshold 2\pi') per node (i.e. dim(array)=params.n)
    :rotation: identifies in which direction we have to descent to count until next max
    
    OUTPUT:
    :avg_count: average amount of nodes that are necessary until the next full phase transition starts
    """

    # Find peaks
    peaks, _ = find_peaks(array)

    alls = np.zeros(len(peaks))

    for idx, max_arg in enumerate(peaks):

        max_arg = np.argmax(array)
    
        count = 0
        node = max_arg
        if rotation<0:
            while array[node - 1] < array[node]:
                count += 1
                node = int(node-1)
        else:
            if node == len(array)-1:
                node == -1
            while array[(node + 1) % len(array)] < array[node]:
                count += 1
                node = (node + 1) % len(array)
        alls[idx] = count

    avg_count = np.mean(alls)
            
    
    return avg_count, peaks

def hilbert_trafo_nd(signal, axis=0):
    """simply the call of the off-shelf implementation to not have to calculate it for every feature individually.
    INPUT:
    :signal: (n,m)-dimensional array of real-valued signal. 
    We have activity=(rows,columns)=(time,nodes) -> default-axis=0.
    
    :output: (n,m)-dimensional array analytical representation of signal
    """
    
    #compute Hilbert Transform for analytical signal representation
    #ue.shape = (time-steps+1, number of nodes)
    #i.e. rows=time, columns=node -> want hilbert trafo w.r.t. time => axis=0
    ana_signal = hilbert(signal, axis=axis)
    
    return(ana_signal)

def hilbert_trafo_1d(signal):
    """simply the call of the off-shelf implementation to not have to calculate it for every feature individually"""
    
    ana_signal = hilbert(signal)
    
    return(ana_signal)

def inst_phase(signal):
    """Compute the instantaneous phase per time step per node."""
    
    #without unwrapping?
    inst_phase = np.unwrap(np.angle(signal))
    
    #inst_phase = np.unwrap(np.angle(signal))
    
    return inst_phase


def inst_frequ(signal):
    """ This function is supposed to determine the instantaneous frequency of a real-valued signal. 
    We use the method from Muller et al (2014), DOI: 10.1038/ncomms4675.
    
    INPUT:
    :signal: analytical representation (a+ib) of real-valued times series, 1-dimensional, array
    
    :output: instantaneous frequency without phase unwrapping, array"""
    
    
    #compute Hilbert Transform for analytical signal representation
    #ue.shape = (time-steps+1, number of nodes)
    #i.e. rows=time, columns=node -> want hilbert trafo w.r.t. time => axis=0
    
    complex_conj = np.conj(signal)
    #roll complex conjugate s.t. in product we compute 
    #(X_n\cdotX^*_{n+1} i.e. we multiply the analytical representation of signal 
    #at space step n with the complex conjugate of the next space step n+1)
    complex_conj = np.roll(complex_conj, -1)
    
    #use elementwise multiplication
    inst_frequ_temp = np.angle(np.multiply(signal, complex_conj))
    
    #omit last one, since it would be the product of X_n\cdotX^*_0
    inst_frequ = inst_frequ_temp[:-1]
    
    return inst_frequ



# # # - - END of PHASE LATENCY functions - - - # # #

# # #  - - - KURAMOTO-ORDER PARAMETER - - - # # #

def getKuramotoOrder(activity, axis=1):
    """
    Determine the Kuramoto order parameter per time step t for a TxN-dimensional activity array of T many time steps and N many nodes.

    INPUT:
    :activity: (T x N)-dimensional array of T time steps of N nodes
    :axis: want to get analytical signal (Hilbert Trafo) PER time step OVER all nodes with default axis=1 (=0 otherwise).

    OUTPUT:
    :kuramoto: T-dimensional array of the kuramoto order parameter R(t) per time step
    :kuramoto_tendency: T-dimensional array of the central tendency of the phases of all oscillators
    """

    analytical_signal = hilbert_trafo_nd(activity, axis=axis)
    theta = np.angle(analytical_signal)

    N = activity.shape[1]
    sums = (1 / N) * np.sum(np.exp(1j*theta), axis=axis)

    kuramoto = np.abs(sums)
    kuramoto_tendency = np.angle(sums)

    return kuramoto, kuramoto_tendency

# # #  - - - KURAMOTO-ORDER PARAMETER END - - - # # #

# # #  - - - EQUIDISTANCY - - - # # #

def getEquidistancy(activity):

    """
    Determine the distances in number of nodes for an N-dimensional activity array of N many nodes.

    INPUT:
    :activity: (T x N)-dimensional array of T time steps of N nodes

    OUTPUT:
    :distances: array of the differences in peaks
    :avg_distance: average distance, float
    :std_distance: standard deviation over distances, Bessel corrected, float
    """

    peaks = find_peaks(activity)[0]
    
    distances = [peaks[i+1]-peaks[i] for i in range(len(peaks)-1)]
    avg_distance = np.mean(distances)
    std_distance = np.std(distances, ddof=1)

    return distances, avg_distance, std_distance

# # #  - - - EQUIDISTANCY END - - - # # #

# # # # # - - - - - Below you can find the two functions rateBatches1d, rateBatchesNd, that determine the average rate of change of an activity u - - - - - # # # # #
# # # - - - IMPORTANT NOTE: rateBatches1d can also be generally used to compute the avg. finite differences in an array between index and index+batch_size - - - # # #
# # # - - - IMPORTANT NOTE: Thi also applies to rateBatchesNd for n-dimensional arrays - - - # # #

# # as of right now, that change in activity isn't too interesting or anything.


def rateBatches1d(array, batch_size=100):
    """"
    This function computes the average rate of change of activity of the activity-array u over space and over time.
    It does so by taking the finite difference of each activity entry u_ij at position i, time j, to u_i(j+batch_size) w.r.t. \Delta t
    INPUT:
    :u: n-dim with shape=(#timesteps, #nodes) activity array (numpy), important note: the transient time should be cut of, otherwise it affects the result.
    :batch_size: integer, how large the batch is that you take for the finite-differences

    OUTPUT:
    :avg_rate: float, average rate of change in activity in u over time and space

    """
    batch_number = int(len(array)/batch_size)
    rate_per_batch = np.zeros(batch_number)
    
    for i in range(batch_number):
        rate_per_batch[i] = np.abs((array[int(i+batch_size)] - array[i]) / batch_size)
        
    avg_rate = np.mean(rate_per_batch)
    
    return avg_rate

def rateBatchesNd(u, batch_size=100):
    """"
    This function computes the average rate of change of activity of the activity-array u over space and over time.
    It does so by taking the finite difference of each activity entry u_ij at position i, time j, to u_i(j+batch_size) w.r.t. \Delta t
    INPUT:
    :u: n-dim with shape=(#timesteps, #nodes) activity array (numpy), important note: the transient time should be cut of, otherwise it affects the result.
    :batch_size: integer, how large the batch is that you take for the finite-differences

    OUTPUT:
    :avg_rate: float, average rate of change in activity in u over time and space
    """

    #IMPORTANT: the array u is already with cut-off transient time!!!
    node_number = u.shape[-1] #to identify, how many nodes I have in my space
    rate_batches_per_node = np.zeros(node_number)
    
    for i in range(node_number):
        rate_batches_per_node[i] = rateBatches1d(u.T[i], batch_size)
        
   # print(rate_batches_per_node)
        
    avg_rate = np.mean(rate_batches_per_node)
    
    return avg_rate