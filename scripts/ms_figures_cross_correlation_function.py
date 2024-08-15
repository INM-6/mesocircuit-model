""" Cross-correlation function
------------------------------


"""

###############################################################################
# Import the necessary modules.

import os
import sys
import numpy as np
import time
import h5py
import subprocess
import mesocircuit
import mesocircuit.mesocircuit_framework as mesoframe
from mesocircuit.helpers.io import load_h5_to_sparse_X
from joblib import Parallel, delayed
import scipy.signal as ss

name = 'crosscorrfunc'


def write_jobscripts(circuit):
    """
    """
    print('Writing jobscripts for cross-correlation functions.')

    name = 'crosscorrfunc'

    # define machine specifics
    for machine in ['hpc', 'local']:
        # start jobscript
        jobscript = '#!/bin/bash -x\n'

        if machine == 'hpc':
            dic = circuit.sys_dict['hpc']['analysis_and_plotting']
            stdout = os.path.join(circuit.data_dir_circuit,
                                  'stdout', name + '.txt')

            jobscript += f"""#SBATCH --job-name=meso_ccf
#SBATCH --partition={dic['partition']}
#SBATCH --output={stdout}
#SBATCH --error={stdout}
#SBATCH --nodes=1
#SBATCH --time=12:00:00
unset DISPLAY
"""

        run_path = os.path.join(os.path.dirname(
            mesocircuit.__file__), '../scripts')
        jobscript += "set -o pipefail\n"
        jobscript += f"RUN_PATH={run_path}\n"
        jobscript += f"DATA_DIR={circuit.data_dir}\n"
        jobscript += f"NAME_EXP={circuit.name_exp}\n"
        jobscript += f"PS_ID={circuit.ps_id}\n"

        # executable
        jobscript += 'python3 -u $RUN_PATH/ms_figures_cross_correlation_function.py $DATA_DIR $NAME_EXP $PS_ID'

        # redirect stdout for local
        if machine == 'local':
            jobscript += f' 2>&1 | tee $DATA_DIR/$NAME_EXP/$PS_ID/stdout/{name}.txt'

        # write jobscript
        fname = os.path.join(circuit.data_dir_circuit, 'jobscripts',
                             f"{machine}_{name}.sh")
        with open(fname, 'w') as f:
            f.write(jobscript)
    return


def run_job(circuit, machine='hpc'):
    """
    Submit jobscript.
    """
    fname = os.path.join(circuit.data_dir_circuit, 'jobscripts',
                         f"{machine}_{name}.sh")

    if machine == 'hpc':
        submit = f'sbatch --account $BUDGET_ACCOUNTS {fname}'
        output = subprocess.getoutput(submit)
        print(output, submit)
    elif machine == 'local':
        retval = os.system(f'bash {fname}')
        if retval != 0:
            raise Exception(f'os.system failed: {retval}')
    return


def compute_cross_correlation_functions(
        num_trains=512, binsize_time_resampled=2, lag_max=25., num_jobs=1):
    """
    Compute pairwise spike trains ignoring autocorrelations.
    """
    start = time.time()

    circuit = mesoframe.Mesocircuit(
        data_dir=sys.argv[-3], name_exp=sys.argv[-2], ps_id=sys.argv[-1],
        load_parameters=True)

    populations = circuit.net_dict['populations'][:-1]  # no TC
    time_step = circuit.ana_dict['binsize_time']
    time_interval = [circuit.ana_dict['t_transient'],
                     circuit.ana_dict['t_transient'] + circuit.sim_dict['t_sim']]
    time_indices = np.arange(time_interval[0] / time_step,
                             time_interval[1] / time_step).astype(int)

    # number of bins to sum for temporally resampling
    q = int(binsize_time_resampled / time_step)

    # load and resample binned spike trains
    fname = os.path.join(
        circuit.data_dir_circuit, 'processed_data', 'all_sptrains_bintime.h5')
    sptrains = {}
    for X in populations:
        with h5py.File(fname, 'r') as f:
            data = load_h5_to_sparse_X(X, f)

        # extract number of spike trains and slice according to time interval
        data = data[:num_trains, time_indices]
        data = data.toarray().astype(float)

        # downsample summing every q column
        data = data.reshape(
            (data.shape[0], data.shape[1] // q, q)).sum(axis=-1)

        # normalize data by subtracting mean and dividing by standard deviation
        data = ((data.T - data.mean(axis=1)) / data.std(axis=1)).T

        sptrains[X] = data

    # lag indices (computed with last data)
    mode = 'same'
    # lags in units of q
    lags = ss.correlation_lags(
        data.shape[1], data.shape[1], mode=mode)
    # convert to usual time units (ms)
    lags *= q
    lag_inds = np.where((lags >= -lag_max) & (lags <= lag_max))

    # compute spike correlations and write them to file
    f = h5py.File(os.path.join(
        circuit.data_dir_circuit, 'processed_data',
        'all_cross_correlation_functions.h5'), 'w')

    print('Computing spike correlations for:')
    for X, Y in zip(['L23E', 'L23E', 'L23I',
                     'L4E', 'L4E', 'L4I',
                     'L5E', 'L5E', 'L5I',
                     'L6E', 'L6E', 'L6I'],
                    ['L23E', 'L23I', 'L23I',
                     'L4E', 'L4I', 'L4I',
                     'L5E', 'L5I', 'L5I',
                     'L6E', 'L6I', 'L6I']):
        print(f'    {X}:{Y}')
        spcorrs = np.array(_compute_spike_correlations(
            sptrains[X], sptrains[Y], lag_inds, mode=mode, num_jobs=num_jobs))
        f[f'{X}:{Y}'] = spcorrs

    f['lag_times'] = lags[lag_inds]
    f.close()

    end = time.time()
    print(f'Computing spike correlations took {end - start:.2f} s.')
    return


def _compute_spike_correlations(
        sptrains_X, sptrains_Y, lag_inds, mode='same', num_jobs=1):
    """
    """
    n_trains = sptrains_X.shape[0]
    mask = np.tri(n_trains, k=0) != True  # ignoring auto-correlations

    def corr(ij, sptrains_X, sptrains_Y, lag_inds):
        cross_corr = ss.correlate(
            sptrains_X[ij[0]], sptrains_Y[ij[1]], mode=mode)
        return cross_corr[lag_inds] / sptrains_X[ij[0]].size

    # use multiprocessing for parallelization
    spcorrs = Parallel(n_jobs=num_jobs)(delayed(corr)(
        ij, sptrains_X, sptrains_Y, lag_inds) for ij in np.c_[np.where(mask)])

    return np.array(spcorrs)


if __name__ == '__main__':
    compute_cross_correlation_functions(
        num_trains=512, binsize_time_resampled=0.5, lag_max=25., num_jobs=64)
