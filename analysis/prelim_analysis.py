import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pickle
import nest

def evaluate(param_path, raster_plot_interval, firing_rates_interval):
    """ Displays simulation results.
    Creates a spike raster plot.
    Calculates the firing rate of each population and displays them as a
    box plot.

    Parameters
    ----------
    raster_plot_interval
        Times (in ms) to start and stop loading spike times for raster plot
        (included).
    firing_rates_interval
        Times (in ms) to start and stop lading spike times for computing
        firing rates (included).

    Returns
    -------
        None

    """
    with open(os.path.join(param_path, 'sim_dict.pkl'), 'rb') as f:
            sim_dict = pickle.load(f)
    with open(os.path.join(param_path, 'net_dict.pkl'), 'rb') as f:
            net_dict = pickle.load(f)

    print('Interval to plot spikes: {} ms'.format(raster_plot_interval))
    plot_raster(
        sim_dict['path_raw_data'],
        sim_dict['path_processed_data'],
        'spike_recorder',
        raster_plot_interval[0],
        raster_plot_interval[1],
        net_dict['N_scaling'])
    print('Interval to compute firing rates: {} ms'.format(
        firing_rates_interval))
    firing_rates(
        sim_dict['path_raw_data'],
        sim_dict['path_processed_data'],
        'spike_recorder',
        firing_rates_interval[0],
        firing_rates_interval[1])
    boxplot(sim_dict['path_processed_data'],
            net_dict['populations'])


def plot_raster(path_raw_data, path_processed_data, name, begin, end, N_scaling):
    """ Creates a spike raster plot of the network activity.

    Parameters
    -----------
    path_raw_data
        Path where the spike times are stored.
    path_processed_data
        Path where the raster plot should be saved.
    name
        Name of the spike recorder.
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

    sd_names, node_ids, data = __load_spike_times(path_raw_data, name, begin, end)
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
    plt.savefig(os.path.join(path_processed_data, 'raster_plot.png'), dpi=300)


def firing_rates(path_raw_data, path_processed_data, name, begin, end):
    """ Computes mean and standard deviation of firing rates per population.

    The firing rate of each neuron in each population is computed and stored
    in a .dat file in the directory of the spike recorders. The mean firing
    rate and its standard deviation are printed out for each population.

    Parameters
    -----------
    path_raw_data
        Path where the spike times are stored.
    path_processed_data
        Path where the firing rates should be saved.
    name
        Name of the spike recorder.
    begin
        Time point (in ms) to start calculating the firing rates (included).
    end
        Time point (in ms) to stop calculating the firing rates (included).

    Returns
    -------
    None

    """
    sd_names, node_ids, data = __load_spike_times(path_raw_data, name, begin, end)
    all_mean_rates = []
    all_std_rates = []
    for i, n in enumerate(sd_names):
        senders = data[i]['sender']
        # 1 more bin than node ids per population
        bins = np.arange(node_ids[i, 0], node_ids[i, 1] + 2)
        spike_count_per_neuron, _ = np.histogram(senders, bins=bins)
        rate_per_neuron = spike_count_per_neuron * 1000. / (end - begin)
        np.savetxt(os.path.join(path_processed_data, ('rate' + str(i) + '.dat')),
                   rate_per_neuron)
        # zeros are included
        all_mean_rates.append(np.mean(rate_per_neuron))
        all_std_rates.append(np.std(rate_per_neuron))
    print('Mean rates: {} spikes/s'.format(np.around(all_mean_rates, decimals=3)))
    print('Standard deviation of rates: {} spikes/s'.format(
        np.around(all_std_rates, decimals=3)))


def boxplot(path_processed_data, populations):
    """ Creates a boxblot of the firing rates of all populations.

    To create the boxplot, the firing rates of each neuron in each population
    need to be computed with the function ``firing_rate()``.

    Parameters
    -----------
    path_processed_data
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
            np.loadtxt(os.path.join(path_processed_data,
                                    'rate' + str(i) + '.dat')))

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
    plt.savefig(os.path.join(path_processed_data, 'box_plot.png'), dpi=300)


def __gather_metadata(path_raw_data, name):
    """ Reads names and ids of spike recorders and first and last ids of
    neurons in each population.

    If the simulation was run on several threads or MPI-processes, one name per
    spike recorder per MPI-process/thread is extracted.

    Parameters
    ------------
    path_raw_data
        Path where the spike recorder files are stored.
    name
        Name of the spike recorder, typically ``spike_recorder``.

    Returns
    -------
    sd_files
        Names of all files written by spike recorders.
    sd_names
        Names of all spike recorders.
    node_ids
        Lowest and highest id of nodes in each population.

    """
    # load filenames
    sd_files = []
    sd_names = []
    for fn in sorted(os.listdir(path_raw_data)):
        if fn.startswith(name):
            sd_files.append(fn)
            # spike recorder name and its ID
            fnsplit = '-'.join(fn.split('-')[:-1])
            if fnsplit not in sd_names:
                sd_names.append(fnsplit)

    # load node IDs
    node_idfile = open(os.path.join(path_raw_data, 'population_nodeids.dat'), 'r')
    node_ids = []
    for l in node_idfile:
        node_ids.append(l.split())
    node_ids = np.array(node_ids, dtype='i4')
    return sd_files, sd_names, node_ids


def __load_spike_times(path_raw_data, name, begin, end):
    """ Loads spike times of each spike recorder.

    Parameters
    ----------
    path_raw_data
        Path where the files with the spike times are stored.
    name
        Name of the spike recorder.
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
    sd_files, sd_names, node_ids = __gather_metadata(path_raw_data, name)
    data = {}
    dtype = {'names': ('sender', 'time_ms'),  # as in header
             'formats': ('i4', 'f8')}
    for i, name in enumerate(sd_names):
        data_i_raw = np.array([[]], dtype=dtype)
        for j, f in enumerate(sd_files):
            if name in f:
                # skip header while loading
                ld = np.loadtxt(os.path.join(path_raw_data, f), skiprows=3, dtype=dtype)
                data_i_raw = np.append(data_i_raw, ld)

        data_i_raw = np.sort(data_i_raw, order='time_ms')
        # begin and end are included if they exist
        low = np.searchsorted(data_i_raw['time_ms'], v=begin, side='left')
        high = np.searchsorted(data_i_raw['time_ms'], v=end, side='right')
        data[i] = data_i_raw[low:high]
    return sd_names, node_ids, data
