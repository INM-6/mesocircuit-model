""" PyNEST Mesocircuit: Stimulus Parameters
-------------------------------------------

A dictionary with parameters for an optional external transient stimulation.
Thalamic input and DC input can be switched on individually.

"""

import numpy as np

stim_dict = {
    # optional thalamic input
    # options are:
    # False:        no thalamic input
    # 'onlycreate': thalamic neurons are created and connected, but they are not
    #               active
    # 'poisson':    persistent thalamic poisson input for a given duration to
    #               all thalamic neurons
    # 'pulses':     repetitive pulses from stimulating thalamic neurons  in the
    #               center of the network
    'thalamic_input': 'pulses',

    # name of the thalamic population (TC stands for Thalamo-Cortical)
    'th_name': 'TC',
    # general parameters for thalamus
    # number of thalamic neurons of network covering 1mm2
    'num_th_neurons_1mm2': 902,
    # connection probabilities of the thalamus to the different populations
    # (same order as in 'populations' in 'net_dict') of network covering 1mm2
    'conn_probs_th_1mm2':
        np.array([0.0, 0.0, 0.0983, 0.0619, 0.0, 0.0, 0.0512, 0.0196]),
    # mean amplitude of the thalamic postsynaptic potential (in mV),
    # standard deviation will be taken from 'net_dict'
    'PSP_th': 0.15,
    # mean delay of the thalamic input (in ms) # TODO intended?
    'delay_th_mean': 1.5,
    # relative standard deviation of the thalamic delay (in ms)
    'delay_th_rel_std': 0.5,

    # thalamic_input = 'poisson'
    # start of the thalamic input (in ms)
    'th_start': 700.0,
    # duration of the thalamic input (in ms)
    'th_duration': 10.0,
    # rate of the thalamic input (in spikes/s)
    'th_rate': 120.0,

    # thalamic_input = 'pulses'
    # only thalamic neurons within a circle of th_radius are stimulated (in mm)
    'th_radius': 0.3,
    # time of first pulse (in ms)
    'th_pulse_start': 700.0,
    # pulse interval (in ms)
    'th_interval': 100.0,
    # delay between the pulse spike generator and the thalamic neurons
    'th_delay_pulse_generator': 1.0,


    # optional DC input
    # turn DC input on or off (True or False)
    'dc_input': False,
    # start of the DC input (in ms)
    'dc_start': 650.0,
    # duration of the DC input (in ms)
    'dc_dur': 100.0,
    # amplitude of the DC input (in pA); final amplitude is population-specific
    # and will be obtained by multiplication with 'K_ext'
    'dc_amp': 0.3}
