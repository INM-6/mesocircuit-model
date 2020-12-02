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
                self.cell.y = self.cell.y + i * self.side_length
                M += self._get_transformation_matrix()
                self.cell.x = x
                self.cell.y = y

        return M
