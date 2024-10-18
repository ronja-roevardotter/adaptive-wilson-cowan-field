import numpy as np
import pandas as pd

import sys
import os

cdir = os.getcwd() #get current directory
os.chdir(cdir) #make sure you are in the current directory
sys.path.append(cdir) #append directory to enable import of functions from current directory files

from integration import run
from funcs import getAvgPSD


# # # # # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # # # # # # # #
# # # - - - - - - - - - - - - - - - - - - - - naive Ansatz to identify pattern type - - - - - - - - - - - - - - - - - - # # #
# # # # # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # # # # # # # #


def collectPatterns(fp, params, maxfreq=300, nperseg=1, ue=None):
    
    """ This function collects the type of activity-pattern that is shown after running a simulation for different settings of parameters 
    (fix given by params, varied in trng-df DataFrame) initialized in each available fixed point per parametrization. 
    Pattern-Identification on basis of frequency over space and over time.
    
    INPUT:
    :fp: fixed point for initialisation 
    :params: dictionary of fix parameters
    :max_freq: integer of maximal frequency to return
    :nperseg: window-size for average PSD computation
    :ue: already (transient time) cut-off activity array of params.n many nodes
    
    OUTPUT:
    :pattern: type of the emerging pattern after initialising the model in the corresponding fixed point.
        stationary=1
        temporal=2
        spatial=3
        spatiotemporal=4
        e.g. parametrization shows 3 fixed points, [fp1, fp2, fp3], init in fp1 shows spatial, in fp2 &fp3 stationary patterns => patterns=[3,1,1]
    :temporal_frequency, spatial_frequency: dominant temporal/spatial frequency of ue (where the corrsponding PSD has maximum power), floats
    :temporal_frequency_std, spatial_frequency_std: standard deviation of dominant temporal/spatial frequency of ue, floats
    """
        
    if np.any(ue)!=None:
        ue = ue
    else:
        ue, _ = run(params, itype='rungekutta', fp=fp)

    fs = (1000 / (1/params.dt)) #temporal sampling frequency

    #temporal and spatial frequncies
    space_frequs, space_psd = getAvgPSD(ue.T, fs=params['n'], nperseg=1)
    time_frequs, time_psd = getAvgPSD(ue, fs=fs, maxfreq=maxfreq, nperseg=nperseg)

    #temporal features
    temporal_frequency = time_frequs[np.argmax(time_psd)]
    temporal_frequency_std = np.std(time_psd, ddof=1)
    temporal_threshold = (2*params.dt) / nperseg
    temporally_homogeneous = any((temporal_frequency < temporal_threshold, all(time_psd <= 1e-5)))

    #spatial features
    spatial_frequency = space_frequs[np.argmax(space_psd)]
    spatial_frequency_std = np.std(space_psd, ddof=1)
    spatially_homogeneous = any((spatial_frequency <= 1, all(space_psd <= 1e-5)))
    
    if spatially_homogeneous and temporally_homogeneous:
        pattern = 1
        return pattern, 0, 0, 0, 0
    elif spatially_homogeneous and not temporally_homogeneous:
        pattern = 2
        return pattern, temporal_frequency, temporal_frequency_std, 0, 0
    elif not spatially_homogeneous and temporally_homogeneous:
        pattern = 3
        return pattern, 0, 0, spatial_frequency, spatial_frequency_std
    else:
        pattern = 4
        return pattern, temporal_frequency, temporal_frequency_std, spatial_frequency, spatial_frequency_std