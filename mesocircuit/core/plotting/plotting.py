"""PyNEST Mesocircuit: Plotting Class
-------------------------------------

The Plotting Class defines plotting functions.
Functions starting with 'plot_' plot to a gridspec cell and are used in figures.py.
"""

import os
import warnings
import h5py
import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, MaxNLocator

# initialize MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

class Plotting:
    """ 
    Provides functions to plot the analyzed data.

    All functions that create a figure start with 'fig_'.

    Parameters
    ---------
    sim_dict
        Dictionary containing all parameters specific to the simulation
        (derived from: ``base_sim_params.py``).
    net_dict
         Dictionary containing all parameters specific to the neuron and
         network models (derived from: ``base_network_params.py``).
    stim_dict
        Dictionary containing all parameters specific to the potential stimulus
        (derived from: ``base_stimulus_params.py``
    ana_dict
        Dictionary containing all parameters specific to the network analysis
        (derived from: ``base_analysis_params.py``
    plot_dict
        Dictionary containing all parameters specific to the plotting
        (derived from: ``base_plotting_params.py``

    """

    def __init__(self, sim_dict, net_dict, stim_dict, ana_dict, plot_dict):
        """
        Initializes some class attributes.
        """
        if RANK == 0:
            print('Instantiating a Plotting object.')

        self.sim_dict = sim_dict
        self.net_dict = net_dict
        self.stim_dict = stim_dict
        self.ana_dict = ana_dict
        self.plot_dict = plot_dict

        # update the matplotlib.rcParams
        matplotlib.rcParams.update(self.plot_dict['rcParams'])

        # TODO this is currently the same as in the __init__ of analysis
        # thalamic population 'TC' is treated as the cortical populations
        # presynaptic population names
        # TODO add TC properly
        self.X = self.net_dict['populations'] 
        #self.X = np.append(self.net_dict['populations'], 'TC')
        # postsynaptic population names
        self.Y = self.net_dict['populations']
        # population sizes
        self.N_X = self.net_dict['num_neurons']
        #self.N_X = np.append(self.net_dict['num_neurons', self.net_dict['num_neurons_th'])

        return


    def __load_h5_to_sparse_X(self, X, h5data):
        """
        TODO currently fct variants duplicated in plotting and analysis
        Loads sparse matrix stored in COOrdinate format in HDF5.

        Parameters
        ----------
        X
            Group name for datasets
            'data', 'row', 'col' vectors of equal length
            'shape' : shape of array tuple
        h5data
            Open .h5 file.
        """
        data_X = sp.coo_matrix((h5data[X]['data_row_col'][()][:, 0],
                               (h5data[X]['data_row_col'][()][:, 1],
                                h5data[X]['data_row_col'][()][:, 2])),
                               shape=h5data[X]['shape'][()])
        return data_X.tocsr()


    def plot_raster(self,
        gs,
        populations,
        all_sptrains,
        all_pos_sorting_arrays,
        time_step,
        time_interval,
        sample_step,
        xlabels=True,
        ylabels=True,
        markersize_scale=0.25):
        """
        Plots spike raster to gridspec cell.

        Neurons are sorted according to sorting_axis applied in
        all_pos_sorting_arrays. 

        Parameters
        ----------
        gs
            A gridspec cell to plot into.
        populations
            List of population names.
        all_sptrains
            Open h5 file with all spike trains.
        all_pos_sorting_arrays
            Open h5 file with position sorting arrays.
        time_step
            Time step corresponding to spike trains.
        time_interval
            Time interval to plot.
        sample_step
            Every sample_step'th neuron is shown (default being 1 means that all
            neurons are shown).
        xlabels
            Boolean indicating if x-labels shall be plotted.
        ylabels
            Boolean indicating if y-labels shall be plotted.
        markersize_scale
            Scaling factor for marker size.

        Returns
        -------
        ax
            Axis to put a label to.
        """
        nums_shown = []
        yticks = []
        ax = plt.subplot(gs)   
        for i,X in enumerate(populations):
            data = self.__load_h5_to_sparse_X(X, all_sptrains)

            # slice according to time interval
            time_indices = np.arange(
                time_interval[0] / time_step,
                time_interval[1] / time_step).astype(int)
            data = data[:, time_indices]

            # sort according to spatial axis
            space_indices = all_pos_sorting_arrays[X][()]
            data = data[space_indices, :]

            # subsample if specified
            if sample_step > 1:
                sample_indices = np.zeros(data.shape[0], dtype=bool)
                sample_indices[::sample_step] = True
                data = data[sample_indices, :]
            
            # final number of neurons to be shown
            num_neurons = data.shape[0]

            # get x,y indices and plot
            y, x = np.nonzero(data.toarray())
            ax.plot(x * time_step + time_interval[0],
                    -(np.sum(nums_shown) + y),
                    marker='$.$',
                    markersize=matplotlib.rcParams['lines.markersize'] * markersize_scale,
                    color=self.plot_dict['pop_colors'][i],
                    markeredgecolor='none',
                    linestyle='',
                    rasterized=True)
            nums_shown.append(num_neurons)
            yticks.append(-np.sum(nums_shown) + 0.5 * nums_shown[-1])

        # draw lines to separate populations on top
        for i,X in enumerate(populations[:-1]):
            ax.plot(time_interval, [-np.sum(nums_shown[:i+1])]*2,
                    'k',
                    linewidth=matplotlib.rcParams['axes.linewidth'])

        ax.set_xlim(time_interval[0], time_interval[1])
        ax.set_ylim(-np.sum(nums_shown), 0)

        ax.set_yticks(yticks)

        if xlabels:
            ax.set_xlabel('time (ms)')
        else:
            ax.set_xticklabels([])
        if ylabels:
            ax.set_yticklabels(self.plot_dict['pop_labels'][:len(nums_shown)])
        else:
            ax.set_yticklabels([])
        return ax


    def plot_statistics_overview(self,
        gs, all_rates, all_LVs, all_CCs, all_PSDs):
        """
        TODO
        """
        axes = [0] * 7
        gs_cols = gridspec.GridSpecFromSubplotSpec(1, 12, subplot_spec=gs,
                                                   wspace=0.5)

        ### column 0: boxcharts
        gs_c0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_cols[0,:2],
                                                 hspace=0.5)
        
        # top: rates
        print('  Plotting boxcharts: rates')
        axes[0] = self.plot_boxcharts(gs_c0[0,0],
            all_rates, xlabel='', ylabel=r'$\nu$ (s$^{-1}$)',
            xticklabels=False)
        
        # middle: LVs
        print('  Plotting boxcharts: LVs')
        axes[1] = self.plot_boxcharts(gs_c0[1,0],
            all_LVs, xlabel='', ylabel='LV',
            xticklabels=False)

        # bottom: CCs
        print('  Plotting boxcharts: CCs')
        axes[2] = self.plot_boxcharts(gs_c0[2,0],
            all_CCs, xlabel='', ylabel='CC')

        ### columns 1, 2, 3: distributions

        # bins used in distribution in [0,1]
        bins_unscaled = (np.arange(0, self.plot_dict['distr_num_bins']+1) /
            self.plot_dict['distr_num_bins'])
        
        # left: rates
        print('  Plotting distributions: rates')
        axes[3] = self.plot_layer_panels(gs_cols[0,3:5],
            xlabel=r'$\nu$ (s$^{-1}$)',
            plotfunc=self.__plotfunc_distributions,
            bins=bins_unscaled * self.plot_dict['distr_max_rate'],
            data=all_rates,
            MaxNLocatorNBins=3,
            ylabel='p (a.u.)')

        # middle: LVs
        print('  Plotting distributions: LVs') 
        axes[4] = self.plot_layer_panels(gs_cols[0,5:7],
            xlabel='LV',
            plotfunc=self.__plotfunc_distributions,
            bins=bins_unscaled * self.plot_dict['distr_max_lv'],
            data=all_LVs,
            MaxNLocatorNBins=3)

        # right: CCs
        print('  Plotting distributions: CCs')
        axes[5] = self.plot_layer_panels(gs_cols[0,7:9],
            xlabel='CC',
            plotfunc=self.__plotfunc_distributions,
            bins=2.*(bins_unscaled-0.5) * self.plot_dict['distr_max_cc'],
            data=all_CCs,
            MaxNLocatorNBins=2)

        ### column 4: PSDs
        print('  Plotting PSDs.')
        axes[6] = self.plot_layer_panels(gs_cols[0,10:],
            xlabel='f (Hz)', ylabel='PSD (s$^{-2}$/Hz)',
            plotfunc=self.__plotfunc_PSDs,
            data=all_PSDs)
        return axes


    def plot_boxcharts(self, gs, data, xlabel='', ylabel='',
        xticklabels=True):
        """
        TODO
        """
        ax = plt.subplot(gs)
        for loc in ['top', 'right']:
            ax.spines[loc].set_color('none')

        data_plot = []
        for X in self.net_dict['populations']:
            # remove potential NANs
            data_X = data[X][~np.isnan(data[X])]
            data_plot.append(data_X)

        # ignore all warnings, target in particular VisibleDeprecationWarning
        # (could be removed in some cases with np.array(data_plot, dtype=object)) 
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            boxes = ax.boxplot(
                data_plot,
                labels=self.plot_dict['pop_labels'][:-1],
                sym='', showmeans=True, patch_artist=True,
                meanprops={'mec' : 'white',
                        'marker' : '_',
                        'markersize' : \
                            matplotlib.rcParams['lines.markersize']*0.5},
                medianprops={'color' : 'k'},
                whiskerprops={'color' : 'k', 'linestyle' : '-'})

            for i,box in enumerate(boxes['boxes']):
                box.set_color(self.plot_dict['pop_colors'][i])

        plt.xticks(rotation=90)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if not xticklabels:
            ax.set_xticklabels([])
        
        ax.yaxis.set_major_locator(MaxNLocator(3))
        return ax


    def plot_layer_panels(self, gs, plotfunc, xlabel='', ylabel='', **kwargs):
        """
        Generic function to plot four vertically arranged panels, one for each
        layer, iterating over populations.

        TODO
        """
        gs_c = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs)#, hspace=0.5)

        layer_count = 0
        for i,X in enumerate(self.net_dict['populations']):
            # select subplot
            if i > 0 and i % 2 == 0:
                layer_count += 1
            if i % 2 == 0:
                ax = plt.subplot(gs_c[layer_count])
                for loc in ['top', 'right']:
                    ax.spines[loc].set_color('none')

            # specific plot
            plotfunc(ax, X, i, **kwargs)

            # ylim
            if i % 2 == 0:
                ymin, ymax = ax.get_ylim()
            if i % 2 == 1:
                ymin1, ymax1 = ax.get_ylim()

                if ax.get_yscale()=='log':
                    y0 = np.min([ymin, ymin1])
                    ax.set_yticks([10.**x for x in np.arange(-10, 10)])
                else:
                    y0 = 0

                ax.set_ylim(y0, np.max([ymax, ymax1]) * 1.1)
                    
            if layer_count == len(self.plot_dict['layer_labels']) - 1:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xticklabels([])

            if i == 0:
                ax.set_ylabel(ylabel)
                ax_label = ax
        return ax_label


    def __plotfunc_distributions(self, ax, X, i, bins, data, MaxNLocatorNBins):
        """
        TODO
        """
        ax.hist(data[X], bins=bins, density=True,
                histtype='step', linewidth=matplotlib.rcParams['lines.linewidth'],
                color=self.plot_dict['pop_colors'][i])

        ax.set_xlim(bins[0], bins[-1])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=MaxNLocatorNBins))
        ax.set_yticks([])
        return 

    
    def __plotfunc_PSDs(self, ax, X, i, data):
        """
        TODO ax limits and ticklabels
        """
        if data[X].size == 0:
            return

        freq, Pxx = data[X]
        # skip frequency of 0 Hz in loglog plot
        freq = freq[1:]
        Pxx = Pxx[1:]
        ax.loglog(freq, Pxx,
                  linewidth=matplotlib.rcParams['lines.linewidth'],
                  color=self.plot_dict['pop_colors'][i])

        ax.set_xticks([10**x for x in np.arange(1, 6)])
        ax.set_xlim(right=self.plot_dict['psd_max_freq'])
        return

    
    def add_label(self, ax, label, offset=[0,0],
                  weight='bold', fontsize_scale=1.2):
        """
        Adds label to axis with given offset.

        Parameters
        ----------
        ax
            Axis to add label to.
        label
            Label should be a letter.
        offset
            x-,y-Offset.
        weight
            Weight of font.
        fontsize_scale
            Scaling factor for font size.
        """
        label_pos = [0.+offset[0], 1.+offset[1]]
        ax.text(label_pos[0], label_pos[1], label,
                ha='left', va='bottom',
                transform=ax.transAxes,
                weight=weight,
                fontsize=matplotlib.rcParams['font.size'] * fontsize_scale)
        return


    def savefig(self, filename, eps_conv=False, eps_conv_via='.svg'):
        """
        Saves the current figure to format given in the plotting parameters.

        TODO: note that inkscape etc. for conversion are not available on JURECA
        
        Parameters
        ----------
        filename
            Name of the file.
        eps_conv
            If the format is .eps and eps_conv=True, the .eps file is converted
            to .pdf and back to .eps to properly compress rasterized parts of
            the figure.
            This is slow but gives a good result with small file size.
        eps_conv_via
            Options are '.svg' (using inkskape) and '.pdf' (using epstopdf and
            pdftops).
        """

        path_fn = os.path.join(self.sim_dict['path_plots'], filename)

        if self.plot_dict['extension'] == '.eps' and eps_conv:

            if eps_conv_via=='.svg':
                prior_ext = '.svg'
                plt.savefig(path_fn + prior_ext)
                cmd = ('inkscape ' + path_fn + '.svg ' +
                       '-E ' + path_fn + '.eps ' +
                       '--export-ignore-filters --export-ps-level=3' + '\n' +  
                       'rm ' + path_fn + '.svg')
                os.system(cmd)

            elif eps_conv_via=='.pdf':
                prior_ext = '.eps'
                plt.savefig(path_fn + prior_ext)
                cmd = ('epstopdf ' + path_fn + '.eps' + '\n' +
                       'pdftops -eps ' + path_fn + '.pdf' + '\n' +
                       'rm ' + path_fn + '.pdf')
                os.system(cmd)

        else:
            plt.savefig(path_fn + self.plot_dict['extension'])

        plt.close()
        return


