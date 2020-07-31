"""PyNEST Mesocircuit: Plotting Parameters
------------------------------------------

A dictionary with parameters defining the plotting.

"""
import numpy as np

# conversion factor from cm to inch
cm2inch = 1. / 2.54

plot_dict = {
    # machine to run the analysis on. Options are 'local' and 'jureca'.
    'computer': 'jureca',

    # number of compute nodes (ignored if computer is 'local')
    'num_nodes': 1,
    # number of MPI processes per compute node
    'num_mpi_per_node': 2,

    # rcParams to overwrite the default ones
    'rcParams': {
        # dpi for typical journal printing
        'figure.dpi': 300,
    },
    # figure width for the J Neurosci (in inch):
    # 1 column, 1.5 columns and 2 columns
    'fig_width_1col': 8.5 * cm2inch,
    'fig_width_15col': 11.6 * cm2inch,
    'fig_width_2col': 17.6 * cm2inch,

    # layer labels
    'layer_labels': np.array(['L2/3', 'L4', 'L5', 'L6']),
    # population labels
    'pop_labels': np.array(['L2/3E', 'L2/3I',
                            'L4E', 'L4I',
                            'L5E', 'L5I',
                            'L6E', 'L6I',
                            'TC']),
    # population colors
    'pop_colors': np.array(['#114477',   # L23E blue
                            '#77AADD',   # L23I
                            '#117744',   # L4E green
                            '#88CCAA',   # L4I
                            '#774411',   # L5E brown
                            '#DDAA77',   # L5I
                            '#771155',   # L6E pompadour
                            '#CC99BB',   # L6I
                            '#696969']), # TC  dimgrey
    # neuron type colors
    'type_colors': np.array(['#595289',   # E, blue pastel
                             '#AF143C',   # I, red pastel
                             '#696969']), # other, dimgrey

}
