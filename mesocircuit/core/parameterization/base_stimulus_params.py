""" PyNEST Mesocircuit: Stimulus Parameters
-------------------------------------------

A dictionary with parameters for an optional external transient stimulation.
Thalamic input and DC input can be switched on individually.

"""

import numpy as np

stim_dict = {
    # optional thalamic input
    # options are:
    # False:        thalamic neurons are created and connected, but they are not
    #               active
    # 'poisson':    persistent thalamic poisson input for a given duration to
    #               all thalamic neurons
    # 'pulses':     repetitive pulses from stimulating thalamic neurons  in the
    #               center of the network
    'thalamic_input': False,

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
