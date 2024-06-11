# Functions for the noise tagging condition of the experiment 1 
import scipy
import numpy as np
import copy
import pandas as pd
import warnings

# Ignore specific RuntimeWarning
warnings.filterwarnings("ignore", message="tmax is not in time interval. tmax is set to")

def normalize_per_trial(data):
    '''
    for each trial and electrode, regularize each time step value
    '''
    return (data-data.mean(axis=(-1), keepdims=True))/data.std(axis=(-1), keepdims=True)

def generate_noise_periods(start, end, duration, step):
    '''
    Select noise tag portions with a rolling window (returned as a list of tupples)
    start and end : part of the tag in which the rolling window is applied
    duration : time length of the window
    step : time between two window starts
    eg : start = 1, end = 2, duration = 0.1, step = 0.01 returns a list of tag portions, of duration 100ms, sampled every 10 ms between 1 and 2s of the tag
    '''
    noise_periods = []
    current_start = start
    
    while current_start + duration <= end:
        current_end = current_start + duration
        noise_periods.append((current_start, current_end))
        current_start += step  # Move to the next interval start
    
    return noise_periods


def cross_cor_nois(epochs_fun, noise, display_side_cued_tag, sampling_freq, electrodes=None,
                  tmin_eeg=0, tmax_eeg=1, tmin_noise=0, tmax_noise=1, mean_tags=False):
    '''
    noise_tags : array trial*tag*timepoint
    display_side_cued_tag : 'left' or 'right'
    sampling_freq : 1920 or 480(Hz)
    tmin_noise : >=0
    mean_tags : if False the cross-correlation is computed separately for the cued and uncued tags, 
                if True only one cross-correlation array is computed for the mean of the tags
    
    Returns 2 matrices (trial*electrode*lag (in time steps between EEG start and tag start) of corr values. 
    The first matrice contains the EEG/cued_tag corr, the second the EEG/tag2 corr (if mean_tags the second matrix is filled with 0s)
    Returns also the trial indexes and the time lag in seconds 
    '''
    display_side_cued_tag = 0 if display_side_cued_tag == 'left' else \
        (1 if display_side_cued_tag == 'right' else ValueError("Invalid value for display_side_cued_tag"))

    # Select elec
    if electrodes is None : 
        eeg = epochs_fun.copy()
    else : eeg = epochs_fun.copy().pick(electrodes)
    noise_tags = copy.deepcopy(noise)

    # Converge EEG and tag sampling frequencies 
    if sampling_freq == 1920:
            eeg.resample(sfreq = sampling_freq)
            # Upsample tag from 480 to 1920Hz (by repetition rather than inference)
            noise_tags = np.repeat(noise_tags, 4, axis=2)
    
    elif sampling_freq == 480:
        eeg.resample(sfreq = sampling_freq)
    
    else : raise ValueError('Sampling frequency must be 1920 or 480Hz')
    
    
    # bandpass (60-80Hz) EEG data + crop from tmin to tmax
    eeg.crop(tmin_eeg, tmax_eeg, verbose=False)
    
    selection = eeg[f'cued_side == {display_side_cued_tag}'].selection # trial indexes
    eeg_arr = eeg[f'cued_side == {display_side_cued_tag}'].get_data(copy=False)

    # Convert tmin/max noise to sample scale
    tmin_noise = int(tmin_noise * sampling_freq)
    tmax_noise = int(tmax_noise * sampling_freq)
    
    # Select the tags on the same trials as the EEG and only the part of between tmin_noise and tmax_noise
    noise_tags = noise_tags[eeg.metadata['trial_number'][selection].values,:,tmin_noise:tmax_noise+1] 

    # Standardize eeg trial data and noise
    eeg_arr = normalize_per_trial(eeg_arr)
    noise_tags = normalize_per_trial(noise_tags)
    
    # Loop over trials and electrodes to fill correlation matrices
    # Cross correlation is done in valid mode so the resulting shape in samples/times is len(eeg) - len(noise) + 1
    cross_corr_tag1 = np.zeros((eeg_arr.shape[0], eeg_arr.shape[1], int(eeg_arr.shape[2]-noise_tags.shape[-1]+1)))
    cross_corr_tag2 = np.zeros((eeg_arr.shape[0], eeg_arr.shape[1], int(eeg_arr.shape[2]-noise_tags.shape[-1]+1)))
    
    for trial in range(eeg_arr.shape[0]):
        for elec in range(eeg_arr.shape[1]):
            trial_data = eeg_arr[trial,elec,:]
            trial_cuednoise = noise_tags[trial,display_side_cued_tag,:]
            trial_noncuednoise = noise_tags[trial,1-display_side_cued_tag,:]

            if mean_tags:
                trial_cuednoise = (trial_cuednoise + trial_noncuednoise )/2
                cross_corr_tag2[trial, elec] = 0
            else:
                cross_corr_tag2[trial, elec] = scipy.signal.correlate(trial_data, trial_noncuednoise, \
                                                        mode='valid')
            cross_corr_tag1[trial, elec] = scipy.signal.correlate(trial_data, trial_cuednoise, \
                                                        mode='valid')
            
    return cross_corr_tag1, cross_corr_tag2, selection, eeg.times[:-len(trial_cuednoise)+1]

def bin_to_dataframe(max_vals, name):
    '''
    Store results from the time window analysis
    '''
    np.save('files/'+name, max_vals)
