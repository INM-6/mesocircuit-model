"""PyNEST Mesocircuit: Helper Functions
---------------------------------------

General helper functions.

"""

import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import parameters as ps
import os
import sys
import operator
import pickle
import hashlib
if 'DISPLAY' not in os.environ: # TODO remove together with plots
    import matplotlib
    matplotlib.use('Agg')

import helpers_network_stimulus as helpnet

# default parameters
from base_sim_params import sim_dict
from base_network_params import net_dict
from base_stimulus_params import stim_dict


def evaluate_parameterspaces(
    filename='', paramspace_keys=[], with_base_params=True):
    """
    Evaluates the parameter spaces as specified by the arguments.


    Parameters
    ----------
    filename
        Name of the file <filename>.py defining the parameterspaces to be
        evaluated (without file extension). The paramterspaces need to be in a
        dictionary named 'ps_dicts'.
        If no filename is given, i.e., filename='', parameters are evaluated
        according to the base parameters (default='').
    paramspace_keys
        List of keys of parameterspaces defined in <filename>.py. Providing an
        empty list means that all keys are evaluated (default=[]).
    with_base_params
        Whether to include a parameterspace with only base parameters
        (default=True).

    Returns
    -------
    parameterview
        Dictionary as overview. Keys are names of parameter spaces and values
        are lists of parameter set ids.

    """
    
    ps_dicts = {}
    try:
        f = __import__(filename)
        ps_dicts.update(f.ps_dicts)
    except:
        pass
    if with_base_params:
        ps_dicts.update({'base': {}})

    # parameterspaces built with the parameters module and indexed by
    # paramspace_key
    parameterspaces = {}
    # collection of unique parametersets indexed by ps_id
    parametersets = {}
    # overview of parameterspaces and corresponding ps_ids
    parameterview = {}

    for paramspace_key in sorted(ps_dicts):
        if (len(paramspace_keys)==0 or # all keys
            paramspace_key in paramspace_keys or # selected key(s)
            paramspace_key=='base'): # base parameters if with_base_params
            print(paramspace_key)
            parameterview[paramspace_key]= []

            parameterspaces[paramspace_key] = ps.ParameterSpace({})
            # start with default parameters and update
            for dic,vdic in zip(
                ['sim_dict', 'net_dict', 'stim_dict'],
                [sim_dict, net_dict, stim_dict]):
                parameterspaces[paramspace_key][dic] = dict(vdic) # copy is needed
                if dic in ps_dicts[paramspace_key]:
                    parameterspaces[paramspace_key][dic].update(
                    ps_dicts[paramspace_key][dic])

            for paramset in parameterspaces[paramspace_key].iter_inner():
                ps_id = get_unique_id(paramset)
                if ps_id in sorted(parametersets):
                    print('Skipping {0}, already in job list.'.format(ps_id))
                    pass
                else:
                    print(ps_id)
                    parametersets[ps_id] = paramset
                    parameterview[paramspace_key].append([
                        paramset['sim_dict']['data_path'], ps_id])

                    evaluate_parameterset(ps_id, paramset)

    return parameterview


def evaluate_parameterset(ps_id, paramset):
    """
    Set paths, derive parameters and write jobscripts for this prameter set.

    Parameters
    ----------
    ps_id
        Unique parameter set id.
    paramset
        Parameter set corresponding to ps_id.    

    """
                
    # set paths and create directories for parameters, jobscripts and
    # raw and processed output data
    for dname in ['parameters', 'jobscripts', 'raw_data', 'processed_data']:
        path = os.path.join(paramset['sim_dict']['data_path'], dname, ps_id)
        if not os.path.isdir(path):
            os.makedirs(path) # also creates sub directories
        paramset['sim_dict']['path_' + dname] = path

    # compute dependent network parameters
    paramset['net_dict'], paramset['stim_dict'] = \
        helpnet.derive_dependent_parameters(
            paramset['net_dict'], paramset['stim_dict'])

    # write final parameters to file (TODO consider human-readable .json)
    for dic in ['sim_dict', 'net_dict', 'stim_dict']:
        with open(os.path.join(paramset['sim_dict']['path_parameters'],
            dic + '.pkl'), 'wb') as f:
            pickle.dump(paramset[dic], f)

    
    # write jobscripts
    run_cmd = \
        'python3 run_mesocircuit.py ' + paramset['sim_dict']['path_parameters']

    if paramset['sim_dict']['computer'] == 'local':
        jobscript = (
            '#!/bin/bash -x' + '\n' +
            'mpirun -n ' + str(paramset['sim_dict']['num_mpi_per_node']) + ' ' +
            run_cmd)
    elif paramset['sim_dict']['computer'] == 'jureca':
        raise Exception # TODO add juerca

    with open(os.path.join(paramset['sim_dict']['path_jobscripts'],
        'network.sh'), 'w') as f:
        f.write(jobscript)

    return

def get_unique_id(d):
    """
    Creates a unique hash key for an input dictionary.

    Parameters
    ----------
    d : dict
        E.g., parameter dictionary.

    Returns
    -------
    key : str
        Hash key.
    """
    d_sorted = sort_deep_dict(d)
    string = pickle.dumps(d_sorted)
    key = hashlib.md5(string).hexdigest()
    return key


def sort_deep_dict(d):
    """
    Sorts arbitrarily deep dictionaries into tuples.

    Parameter
    ---------
    d : dict

    Returns
    -------
    x : list of tuples of tuples of tuples ...
    """
    x = sorted(iter(list(d.items())), key=operator.itemgetter(0))
    for i, (key, value) in enumerate(x):
        if type(value) == dict or type(value) == ps.ParameterSet:
            y = sorted(iter(list(value.items())), key=operator.itemgetter(0))
            x[i] = (key, y)
            for j, (k, v) in enumerate(y):
                if type(v) == dict or type(v) == ps.ParameterSet:
                    y[j] = (k, sort_deep_dict(v))
    return x


def plot_raster(path, name, begin, end, N_scaling): # TODO remove
    """ Creates a spike raster plot of the network activity.

    Parameters
    -----------
    path
        Path where the spike times are stored.
    name
        Name of the spike detector.
    begin
        Time point (in ms) to start plotting spikes (included).
    end
        Time point (in ms) to stop plotting spikes (included).
    N_scaling
        Scaling factor for number of neurons.

    Returns
    -------
    None

    """
    fs = 18  # fontsize
    ylabels = ['L2/3', 'L4', 'L5', 'L6']
    color_list = np.tile(['#595289', '#af143c'], 4)

    sd_names, node_ids, data = __load_spike_times(path, name, begin, end)
    last_node_id = node_ids[-1, -1]
    mod_node_ids = np.abs(node_ids - last_node_id) + 1

    label_pos = [(mod_node_ids[i, 0] + mod_node_ids[i + 1, 1]) /
                 2. for i in np.arange(0, 8, 2)]

    stp = 1
    if N_scaling > 0.1:
        stp = int(10. * N_scaling)
        print('  Only spikes of neurons in steps of {} are shown.'.format(stp))

    plt.figure(figsize=(8, 6))
    for i, n in enumerate(sd_names):
        times = data[i]['time_ms']
        neurons = np.abs(data[i]['sender'] - last_node_id) + 1
        plt.plot(times[::stp], neurons[::stp], '.', color=color_list[i])
    plt.xlabel('time [ms]', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(label_pos, ylabels, fontsize=fs)
    plt.savefig(os.path.join(path, 'raster_plot.png'), dpi=300)


def firing_rates(path, name, begin, end):
    """ Computes mean and standard deviation of firing rates per population.

    The firing rate of each neuron in each population is computed and stored
    in a .dat file in the directory of the spike detectors. The mean firing
    rate and its standard deviation are printed out for each population.

    Parameters
    -----------
    path
        Path where the spike times are stored.
    name
        Name of the spike detector.
    begin
        Time point (in ms) to start calculating the firing rates (included).
    end
        Time point (in ms) to stop calculating the firing rates (included).

    Returns
    -------
    None

    """
    sd_names, node_ids, data = __load_spike_times(path, name, begin, end)
    all_mean_rates = []
    all_std_rates = []
    for i, n in enumerate(sd_names):
        senders = data[i]['sender']
        # 1 more bin than node ids per population
        bins = np.arange(node_ids[i, 0], node_ids[i, 1] + 2)
        spike_count_per_neuron, _ = np.histogram(senders, bins=bins)
        rate_per_neuron = spike_count_per_neuron * 1000. / (end - begin)
        np.savetxt(os.path.join(path, ('rate' + str(i) + '.dat')),
                   rate_per_neuron)
        # zeros are included
        all_mean_rates.append(np.mean(rate_per_neuron))
        all_std_rates.append(np.std(rate_per_neuron))
    print('Mean rates: {} spikes/s'.format(np.around(all_mean_rates, decimals=3)))
    print('Standard deviation of rates: {} spikes/s'.format(
        np.around(all_std_rates, decimals=3)))


def boxplot(path, populations):
    """ Creates a boxblot of the firing rates of all populations.

    To create the boxplot, the firing rates of each neuron in each population
    need to be computed with the function ``firing_rate()``.

    Parameters
    -----------
    path
        Path where the firing rates are stored.
    populations
        Names of neuronal populations.

    Returns
    -------
    None

    """
    fs = 18
    pop_names = [string.replace('23', '2/3') for string in populations]
    label_pos = list(range(len(populations), 0, -1))
    color_list = ['#af143c', '#595289']
    medianprops = dict(linestyle='-', linewidth=2.5, color='black')
    meanprops = dict(linestyle='--', linewidth=2.5, color='lightgray')

    rates_per_neuron_rev = []
    for i in np.arange(len(populations))[::-1]:
        rates_per_neuron_rev.append(
            np.loadtxt(os.path.join(path, ('rate' + str(i) + '.dat'))))

    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(rates_per_neuron_rev, 0, 'rs', 0, medianprops=medianprops,
                     meanprops=meanprops, meanline=True, showmeans=True)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # boxcolors
    for i in np.arange(len(populations)):
        boxX = []
        boxY = []
        box = bp['boxes'][i]
        for j in list(range(5)):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        k = i % 2
        boxPolygon = Polygon(boxCoords, facecolor=color_list[k])
        plt.gca().add_patch(boxPolygon)
    plt.xlabel('firing rate [spikes/s]', fontsize=fs)
    plt.yticks(label_pos, pop_names, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.savefig(os.path.join(path, 'box_plot.png'), dpi=300)


def __gather_metadata(path, name):
    """ Reads names and ids of spike detectors and first and last ids of
    neurons in each population.

    If the simulation was run on several threads or MPI-processes, one name per
    spike detector per MPI-process/thread is extracted.

    Parameters
    ------------
    path
        Path where the spike detector files are stored.
    name
        Name of the spike detector, typically ``spike_detector``.

    Returns
    -------
    sd_files
        Names of all files written by spike detectors.
    sd_names
        Names of all spike detectors.
    node_ids
        Lowest and highest id of nodes in each population.

    """
    # load filenames
    sd_files = []
    sd_names = []
    for fn in sorted(os.listdir(path)):
        if fn.startswith(name):
            sd_files.append(fn)
            # spike detector name and its ID
            fnsplit = '-'.join(fn.split('-')[:-1])
            if fnsplit not in sd_names:
                sd_names.append(fnsplit)

    # load node IDs
    node_idfile = open(path + 'population_nodeids.dat', 'r')
    node_ids = []
    for l in node_idfile:
        node_ids.append(l.split())
    node_ids = np.array(node_ids, dtype='i4')
    return sd_files, sd_names, node_ids


def __load_spike_times(path, name, begin, end):
    """ Loads spike times of each spike detector.

    Parameters
    ----------
    path
        Path where the files with the spike times are stored.
    name
        Name of the spike detector.
    begin
        Time point (in ms) to start loading spike times (included).
    end
        Time point (in ms) to stop loading spike times (included).

    Returns
    -------
    data
        Dictionary containing spike times in the interval from ``begin``
        to ``end``.

    """
    sd_files, sd_names, node_ids = __gather_metadata(path, name)
    data = {}
    dtype = {'names': ('sender', 'time_ms'),  # as in header
             'formats': ('i4', 'f8')}
    for i, name in enumerate(sd_names):
        data_i_raw = np.array([[]], dtype=dtype)
        for j, f in enumerate(sd_files):
            if name in f:
                # skip header while loading
                ld = np.loadtxt(os.path.join(path, f), skiprows=3, dtype=dtype)
                data_i_raw = np.append(data_i_raw, ld)

        data_i_raw = np.sort(data_i_raw, order='time_ms')
        # begin and end are included if they exist
        low = np.searchsorted(data_i_raw['time_ms'], v=begin, side='left')
        high = np.searchsorted(data_i_raw['time_ms'], v=end, side='right')
        data[i] = data_i_raw[low:high]
    return sd_names, node_ids, data
