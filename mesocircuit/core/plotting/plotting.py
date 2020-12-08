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
from matplotlib.colors import SymLogNorm
from ..helpers import base_class

# initialize MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

class Plotting(base_class.BaseAnalysisPlotting):
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

        # inherit from parent class
        super().__init__(sim_dict, net_dict, stim_dict, ana_dict)

        # plot_dict is not in parent class
        self.plot_dict = plot_dict

        # update the matplotlib.rcParams
        matplotlib.rcParams.update(self.plot_dict['rcParams'])
        return


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
            data = self.load_h5_to_sparse_X(X, all_sptrains)

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
        gs, all_FRs, all_LVs, all_CCs, all_PSDs):
        """
        TODO
        """
        axes = [0] * 7
        gs_cols = gridspec.GridSpecFromSubplotSpec(1, 12, subplot_spec=gs,
                                                   wspace=0.5)

        ### column 0: boxcharts
        gs_c0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_cols[0,:2],
                                                 hspace=0.5)
        
        # top: FRs
        print('  Plotting boxcharts: rates')
        axes[0] = self.plot_boxcharts(gs_c0[0,0],
            all_FRs, xlabel='', ylabel='FR (spikes/s)',
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
        
        # left: FRs
        print('  Plotting distributions: FRs')
        axes[3] = self.plot_layer_panels(gs_cols[0,3:5],
            xlabel='FR (spikes/s)',
            plotfunc=self.plotfunc_distributions,
            bins=bins_unscaled * self.plot_dict['distr_max_rate'],
            data=all_FRs,
            MaxNLocatorNBins=3,
            ylabel='p (a.u.)')

        # middle: LVs
        print('  Plotting distributions: LVs') 
        axes[4] = self.plot_layer_panels(gs_cols[0,5:7],
            xlabel='LV',
            plotfunc=self.plotfunc_distributions,
            bins=bins_unscaled * self.plot_dict['distr_max_lv'],
            data=all_LVs,
            MaxNLocatorNBins=3)

        # right: CCs
        print('  Plotting distributions: CCs')
        axes[5] = self.plot_layer_panels(gs_cols[0,7:9],
            xlabel='CC',
            plotfunc=self.plotfunc_distributions,
            bins=2.*(bins_unscaled-0.5) * self.plot_dict['distr_max_cc'],
            data=all_CCs,
            MaxNLocatorNBins=2)

        ### column 4: PSDs
        print('  Plotting PSDs.')
        axes[6] = self.plot_layer_panels(gs_cols[0,10:],
            xlabel='f (Hz)', ylabel='PSD (s$^{-2}$/Hz)',
            plotfunc=self.plotfunc_PSDs,
            data=all_PSDs)
        return axes


    def plot_spatial_snapshots(self,
        gs,
        populations,
        all_inst_rates_bintime_binspace,
        binsize_time,
        binsize_space,
        start_time='th_pulse_start', # ms
        step=1, # multiplication
        nframes=8,
        tickstep=2,
        cbar=True,
        cbar_bottom=0.12,
        cbar_height=0.02):
        """
        """
        # if the start time is a string, the respective entry from the stimulus
        # parameters is used
        if type(start_time) == str:
            start_time = self.stim_dict[start_time]

        start_frame = int(start_time / binsize_time) 
        end_frame = start_frame + (nframes - 1) * step
        times = np.arange(start_frame, end_frame+1, step) * binsize_time

        numbins = self.space_bins.size - 1

        # minimum rate a bit below 0 to avoid issues with cmap.set_under()
        vmin = -0.001
        # separator between sub-panels masked with cmap.set_under()
        val_sep = -1

        for X in populations:
            data = self.load_h5_to_sparse_X(X, all_inst_rates_bintime_binspace)
            data = data[:, start_frame:end_frame+1:step].toarray()
            data = data.reshape((numbins, -1, data.shape[-1]))

            # append frames as columns
            separator_frames = np.array([val_sep]*(numbins)).reshape(-1,1)
            data0 = np.concatenate((data[:,:,0], separator_frames), axis=1)
            for n in np.arange(nframes-1):
                data_apnd = np.concatenate((data0, data[:,:,n+1]), axis=1)
                data0 = np.concatenate((data_apnd, separator_frames), axis=1)

            # append populations as rows
            separator_pops = \
                np.array([val_sep] * np.shape(data_apnd)[1]).reshape(1,-1)
            if X==populations[0]:
                plot_data0 = np.concatenate((data_apnd, separator_pops), axis=0)
            else:
                plot_data = np.concatenate((plot_data0, data_apnd), axis=0)
                plot_data0 = np.concatenate((plot_data, separator_pops), axis=0)

        ax = plt.subplot(gs)

        cmap = matplotlib.cm.Greys
        cmap.set_under(color='black')
        im = ax.imshow(plot_data, interpolation='nearest', cmap=cmap,
                       vmin=vmin,
                       vmax=self.plot_dict['snapshots_max_rate'])

        # ticks dependent on number of spatial bins
        xy_ticks = [numbins / 2.]
        for t in np.arange(np.max([nframes - 1, len(self.X) - 1])):
            xy_ticks.append(xy_ticks[-1] + numbins + 1.)

        xticks = xy_ticks[:nframes:tickstep]
        ticklabels = times[::tickstep]
        if (int(ticklabels[0]) == ticklabels[0] and
            int(ticklabels[1]) == ticklabels[1]):
            ticklabels = ticklabels.astype(int)

        ax.set_xticks(xticks)
        ax.set_xticklabels(ticklabels)

        ax.set_yticks(xy_ticks[:len(self.X)])
        ax.set_yticklabels(self.plot_dict['pop_labels'][:len(self.X)])

        ax.set_xlabel('time (ms)')

        if cbar:
            fig = plt.gcf()
            rect = np.array(ax.get_position().bounds)
            rect[0] += 0.0 # left
            rect[2] -= 0.0 # width
            rect[1] -= cbar_bottom # bottom
            rect[3] = cbar_height # height

            cax = fig.add_axes(rect)
            cb = fig.colorbar(
                im, cax=cax, orientation='horizontal', extend='max')
            cax.xaxis.set_label_position('bottom')
            cb.set_label('FR (spikes/s)')
            cb.locator = MaxNLocator(nbins=5)
            cb.update_ticks() # necessary for location

        return ax


    def plot_crosscorrelation_funcs_thalamic_pulses(self,
        gs,
        populations,
        all_CCfuncs_thalamic_pulses,
        wspace=0.2,
        cbar=True,
        cbar_left = 0.4,
        cbar_bottom=0.12,
        cbar_height=0.02):
        """
        """
        ncols = int(np.floor(np.sqrt(len(populations))))
        nrows = len(populations) // ncols
        gsf = gridspec.GridSpecFromSubplotSpec(
            nrows, ncols, subplot_spec=gs, wspace=wspace)

        for i, X in enumerate(populations):
            ax = plt.subplot(gsf[i])

            cmap = 'RdGy_r'
            vmax = 0.5
            vmin = -vmax
            linthresh = 0.05

            cc_func = all_CCfuncs_thalamic_pulses[X]['cc_funcs']
            distances = all_CCfuncs_thalamic_pulses[X]['distances_mm']
            lags = all_CCfuncs_thalamic_pulses[X]['lags_ms']
            dstep = distances[1] - distances[0]

            im = ax.imshow(cc_func, cmap=cmap, aspect='auto',
                           extent=[lags[0], lags[-1],
                                   distances[0]-dstep/2., distances[-1]+dstep/2.],
                        norm=SymLogNorm(linthresh=linthresh, linscale=1,
                                        vmin=vmin, vmax=vmax),
                        interpolation='nearest',
                        origin='lower')
            ax.axis(ax.axis('tight'))
            # grid lines
            ax.grid(which='major', axis='both', linestyle=':', color='k')

            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            layer = self.plot_dict['layer_labels'][int(i/2.)]
            if i==0:
                ax.set_title('E')
                ax.set_ylabel('r (mm)\n' + layer)
                ax_return = ax
            if i % ncols==0 and i!=0:
                ax.set_ylabel(layer)

            if i==1:
                ax.set_title('I')

            if i % ncols > 0:
                ax.set_yticklabels([])

            if i >= len(populations)-2:
                ax.set_xlabel(r'$\tau$ (ms)')
            else:
                ax.set_xticklabels([])

            if cbar:
                if i==len(populations)-1:
                    fig = plt.gcf()
                    rect = np.array(ax.get_position().bounds)
                    rect[0] -= cbar_left # left
                    rect[2] += cbar_left # width
                    rect[1] -= cbar_bottom # bottom
                    rect[3] = cbar_height # height

                    cax = fig.add_axes(rect)
                    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
                    cax.xaxis.set_label_position('bottom')
                    cb.set_label(r'CC$^\mathrm{FR}\, (\tau, r)$', labelpad=0.1)
                    ticks = [vmin, -linthresh, 0, linthresh, vmax]
                    cb.set_ticks(ticks)
                    cb.set_ticklabels(ticks)
                    cb.update_ticks() # necessary for locator
        return ax_return


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


    def plotfunc_distributions(self, ax, X, i, bins, data, MaxNLocatorNBins):
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

    
    def plotfunc_PSDs(self, ax, X, i, data):
        """
        TODO ax limits and ticklabels
        """
        # return if no data
        if type(data[X]) == h5py._hl.dataset.Dataset and data[X].size == 0:
            return

        freq = data[X]['frequencies_s-1']
        Pxx = data[X]['psds_s^-2_Hz-1']

        # skip frequency of 0 Hz in loglog plot
        freq = freq[1:]
        Pxx = Pxx[1:]
        ax.loglog(freq, Pxx,
                  linewidth=matplotlib.rcParams['lines.linewidth'],
                  color=self.plot_dict['pop_colors'][i])

        ax.set_xticks([10**x for x in np.arange(1, 6)])
        ax.set_xlim(right=self.plot_dict['psd_max_freq'])
        return


    def plotfunc_CCs_distance(self,
        ax, X, i, data, max_num_pairs=10000, markersize_scale=0.4, nblocks=3):
        """
        """
        # return if no data
        if type(data[X]) == h5py._hl.dataset.Dataset and data[X].size == 0:
            return

        distances = data[X]['distances_mm'][:max_num_pairs]
        ccs = data[X]['ccs'][:max_num_pairs]

        # loop for reducing zorder-bias
        blocksize = int(len(distances) / nblocks)
        for b in np.arange(nblocks):
            indices = np.arange(b*blocksize, (b+1)*blocksize)
            zorder = 2*b + i%2 # alternating for populations

            ax.plot(distances[indices],
                    ccs[indices],
                    marker='$.$',
                    markersize=matplotlib.rcParams['lines.markersize'] * markersize_scale,
                    color=self.plot_dict['pop_colors'][i],
                    markeredgecolor='none',
                    linestyle='',
                    zorder=zorder,
                    rasterized=True)
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


