#!/usr/bin/env python
'''derived class definitions for LFP predictions'''
import numpy as np
from lfpykit import RecExtElectrode


class PeriodicLFP(RecExtElectrode):
    '''
    Modified version of lfpykit.RecExtElectrode that incorporates periodic
    boundary conditions for electrostatic forward models in 2D.

    Parameters
    ----------
    side_length: float > 0
        periodicity along lateral x and y direction. Default: 4000 µm
    order: int >= 1
        number of "mirror" sources. Default: 1

    **kwargs:
        lfpykit.RecExtElectrode parameters
    '''

    def __init__(self, side_length=4000., order=1, **kwargs):
        """Initialize RecExtElectrode class"""
        super().__init__(**kwargs)
        self._get_transformation_matrix = super().get_transformation_matrix

        self.side_length = side_length
        self.order = order

    def get_transformation_matrix(self):
        '''
        Get linear response matrix

        Returns
        -------
        response_matrix: ndarray
            shape (n_contacts, n_seg) ndarray

        Raises
        ------
        AttributeError
            if `cell is None`
        '''
        if self.cell is None:
            raise AttributeError(
                '{}.cell is None'.format(self.__class__.__name__))
        M = np.zeros((self.x.size, self.cell.totnsegs))
        for i in range(-self.order, self.order + 1):
            for j in range(-self.order, self.order + 1):
                x = self.cell.x.copy()
                y = self.cell.y.copy()
                self.cell.x = self.cell.x + i * self.side_length
                self.cell.y = self.cell.y + j * self.side_length
                M += self._get_transformation_matrix()
                self.cell.x = x
                self.cell.y = y

        return M


class LaminarProbe(PeriodicLFP):
    '''
    LaminarProbe class wrapping PeriodicLFP class

    Parameters
    ----------
    **kwargs
        parameters parsed to class PeriodicLFP
    '''

    def __init__(self, **kwargs):
        '''Initialize LaminarProbe class wrapping PeriodicLFP class'''
        super().__init__(**kwargs)


if __name__ == '__main__':
    '''test sim checking that imaginary sources across periodic boundaries
    are contributing to the EP'''
    import matplotlib.pyplot as plt
    import LFPy

    cell = LFPy.Cell(morphology='morphologies/L6E_p6(L56).hoc',
                     tstop=5, dt=0.05)
    cell.set_pos(x=0, y=0, z=200)

    syn = LFPy.Synapse(cell=cell, idx=0, syntype='Exp2Syn', weight=1.)
    syn.set_spike_times(np.array([1.]))

    # electrode coordinates and params
    x = np.arange(-10000, 10001, 200)
    y = np.zeros(x.size)
    z = np.zeros(x.size)
    n = 50
    r = 10.
    N = [[0., 0., 1.]] * x.size

    # normal prediction without periodic boundaries
    electrode = LFPy.RecExtElectrode(cell=cell, x=x, y=y, z=z,
                                     n=n, r=r, N=N,
                                     method='root_as_point',)
    # periodic boundaries
    periodic = PeriodicLFP(cell=cell, x=x, y=y, z=z,
                           n=n, r=r, N=N,
                           method='root_as_point',
                           order=2,
                           side_length=4000.)
    cell.simulate(probes=[electrode, periodic])

    _, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=False, sharey=True)
    for i, (probe, ax) in enumerate(zip([electrode, periodic], axes)):
        im = ax.pcolormesh(
            cell.tvec, x, probe.data,
            vmin=-abs(electrode.data).max(),
            vmax=abs(electrode.data).max(),
            cmap='RdBu',
            rasterized=True,
            shading='auto')
        ax.axis(ax.axis('tight'))
        if i == 0:
            ax.set_ylabel('cell x position (µm)', labelpad=0)
        ax.set_xlabel('time (ms)', labelpad=0)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('extracellular potential (mV)', labelpad=0)
        ax.set_title(probe.__class__.__name__)
    plt.show()

    del syn, electrode, periodic
    cell.__del__()
