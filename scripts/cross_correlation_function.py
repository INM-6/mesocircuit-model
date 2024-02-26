""" Cross-correlation function
------------------------------


"""

###############################################################################
# Import the necessary modules.

import os
import sys
import subprocess
import mesocircuit
import mesocircuit.mesocircuit_framework as mesoframe
import mesocircuit.analysis.spike_analysis as sana
import mesocircuit.helpers.parallelism_time as pt

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
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
export NUMEXPR_MAX_THREADS={dic['max_num_cores']}
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS={dic['max_num_cores']}
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
        jobscript += 'python3 -u $RUN_PATH/cross_correlation_function.py $DATA_DIR $NAME_EXP $PS_ID'

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


def compute_and_plot_cross_correlation_function():
    """
    """
    circuit = mesoframe.Mesocircuit(
        data_dir=sys.argv[-3], name_exp=sys.argv[-2], ps_id=sys.argv[-1],
        load_parameters=True)

    # TODO continue here
    print(1234)

    return


if __name__ == '__main__':
    compute_and_plot_cross_correlation_function()
