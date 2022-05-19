#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mesocircuit-project test suite.

"""

import os
import sys
import numpy as np
import unittest

# hack until we have a proper python package
sys.path.append(os.path.split(os.getcwd())[0])


class TestSuite(unittest.TestCase):
    """
    test methods and modules
    """

    def test_TestSuite_00(self):
        '''test TestSuite'''
        self.assertTrue(True)

    def test_SpikeAnalysis__time_binned_sptrains_X_00(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 1000
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 1000, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array([X]),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}

        time_bins = np.arange(sim_dict['t_sim'])

        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        sptrains_bin_time = sana._SpikeAnalysis__time_binned_sptrains_X(
            X=X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        self.assertTrue(np.all(sptrains_bin_time.toarray() == np.eye(N_X)))

    def test_SpikeAnalysis__time_binned_sptrains_X_01(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 50
        spikes = np.array(list(zip([1, 1, 2, 4], [1.5, 100.5, 200.5, 250.5])),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 300, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array([X]),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}

        time_bins = np.arange(sim_dict['t_sim'])

        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        sptrains_bin_time = sana._SpikeAnalysis__time_binned_sptrains_X(
            X=X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        gt = np.zeros((N_X, time_bins.size))
        gt[1, 1] = 1
        gt[1, 100] = 1
        gt[2, 200] = 1
        gt[4, 250] = 1

        self.assertTrue(np.all(sptrains_bin_time.toarray() == gt))

    def test_SpikeAnalysis__time_binned_sptrains_X_02(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 50
        spikes = np.array(list(zip([4, 2, 1, 1], [1.5, 100.5, 200.5, 250.5])),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 300, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array([X]),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}

        time_bins = np.arange(sim_dict['t_sim'])

        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        sptrains_bin_time = sana._SpikeAnalysis__time_binned_sptrains_X(
            X=X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        gt = np.zeros((N_X, time_bins.size))
        gt[4, 1] = 1
        gt[2, 100] = 1
        gt[1, 200] = 1
        gt[1, 250] = 1

        self.assertTrue(np.all(sptrains_bin_time.toarray() == gt))

    def test_SpikeAnalysis__time_binned_sptrains_X_03(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 50
        spikes = np.array(list(zip([1, 4, 2, 1], [100.5, 1.5, 250.5, 200.5])),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 300, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array([X]),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}

        time_bins = np.arange(sim_dict['t_sim'])

        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        sptrains_bin_time = sana._SpikeAnalysis__time_binned_sptrains_X(
            X=X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        gt = np.zeros((N_X, time_bins.size))
        gt[4, 1] = 1
        gt[1, 100] = 1
        gt[1, 200] = 1
        gt[2, 250] = 1

        self.assertTrue(np.all(sptrains_bin_time.toarray() == gt))

    def test_SpikeAnalysis_time_and_space_binned_sptrains_X_00(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one centered unit per spatial bin
        x, y = np.meshgrid(np.arange(-5, 5) + 0.5, np.arange(-5, 5) + 0.5)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 100, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array(['E']),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}

        time_bins = np.arange(sim_dict['t_sim'])

        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        sptrains_bin_time = sana._SpikeAnalysis__time_binned_sptrains_X(
            X=X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        sptrains_bin_space_time = sana._time_and_space_binned_sptrains_X(
            X, positions, sptrains_bin_time,
            dtype=np.uint16)

        gt = np.eye((sana.space_bins.size - 1)**2)

        self.assertTrue(np.all(sptrains_bin_space_time.toarray() == gt))

    def test_SpikeAnalysis_time_and_space_binned_sptrains_X_01(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one unit per spatial bin located on left-hand edge
        # (we count intervals [LH, RH>)
        x, y = np.meshgrid(np.arange(-5, 5), np.arange(-5, 5))
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 100, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array(['E']),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}

        time_bins = np.arange(sim_dict['t_sim'])

        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        sptrains_bin_time = sana._SpikeAnalysis__time_binned_sptrains_X(
            X=X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        sptrains_bin_space_time = sana._time_and_space_binned_sptrains_X(
            X, positions, sptrains_bin_time,
            dtype=np.uint16)

        gt = np.eye((sana.space_bins.size - 1)**2)

        self.assertTrue(np.all(sptrains_bin_space_time.toarray() == gt))

    def test_SpikeAnalysis_time_and_space_binned_sptrains_X_02(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one unit per spatial bin located near right-hand edge
        # (we count intervals [LH, RH>)
        x, y = np.meshgrid(np.arange(-5, 5) + 0.9999,
                           np.arange(-5, 5) + 0.9999)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 100, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array(['E']),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}

        time_bins = np.arange(sim_dict['t_sim'])

        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        sptrains_bin_time = sana._SpikeAnalysis__time_binned_sptrains_X(
            X=X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        sptrains_bin_space_time = sana._time_and_space_binned_sptrains_X(
            X, positions, sptrains_bin_time,
            dtype=np.uint16)

        gt = np.eye((sana.space_bins.size - 1)**2)

        self.assertTrue(np.all(sptrains_bin_space_time.toarray() == gt))

    def test_SpikeAnalysis_time_and_space_binned_sptrains_X_03(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one unit per spatial bin located on right-hand edge
        # (we count intervals [LH, RH>)
        x, y = np.meshgrid(np.arange(-5, 5) + 1, np.arange(-5, 5) + 1)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 100, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array(['E']),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}

        time_bins = np.arange(sim_dict['t_sim'])

        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        sptrains_bin_time = sana._SpikeAnalysis__time_binned_sptrains_X(
            X=X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        try:
            sptrains_bin_space_time = sana._time_and_space_binned_sptrains_X(
                X, positions, sptrains_bin_time,
                dtype=np.uint16)
        except NotImplementedError:
            pass

    def test_SpikeAnalysis_time_and_space_binned_sptrains_X_04(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one unit per spatial bin located on right-hand edge
        # (we count intervals [LH, RH>)
        x, y = np.meshgrid(np.arange(-5, 5) - 0.0001,
                           np.arange(-5, 5) - 0.0001)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 100, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array(['E']),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}

        time_bins = np.arange(sim_dict['t_sim'])

        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        sptrains_bin_time = sana._SpikeAnalysis__time_binned_sptrains_X(
            X=X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        try:
            sptrains_bin_space_time = sana._time_and_space_binned_sptrains_X(
                X, positions, sptrains_bin_time,
                dtype=np.uint16)
        except NotImplementedError:
            pass

    def test_SpikeAnalysis_time_and_space_binned_sptrains_X_05(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 100
        spikes = np.array(list(zip(np.hstack([np.arange(N_X)] * 3), np.arange(N_X * 3) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', float)])
        spikes = spikes[spikes['time_ms'].argsort()]

        # one centered unit per spatial bin
        x, y = np.meshgrid(np.arange(-5, 5) + 0.5, np.arange(-5, 5) + 0.5)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        # three spikes per neuron per spatial bin
        time_bins = np.arange(N_X * 3)

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0,
                    't_sim': time_bins.size,
                    'sim_resolution': 0.1}
        net_dict = {'populations': np.array(['E']),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}

        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        sptrains_bin_time = sana._SpikeAnalysis__time_binned_sptrains_X(
            X=X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        sptrains_bin_space_time = sana._time_and_space_binned_sptrains_X(
            X, positions, sptrains_bin_time,
            dtype=np.uint16)

        gt = np.c_[np.eye((sana.space_bins.size - 1)**2),
                   np.eye((sana.space_bins.size - 1)**2),
                   np.eye((sana.space_bins.size - 1)**2)]

        self.assertTrue(np.all(sptrains_bin_space_time.toarray() == gt))

    def test_SpikeAnalysis_time_and_space_binned_sptrains_X_06(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 100
        spikes = np.array(list(zip([1, 1, 2, 4], [1.5, 100.5, 200.5, 250.5])),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one centered unit per spatial bin
        x, y = np.meshgrid(np.arange(-5, 5) + 0.5, np.arange(-5, 5) + 0.5)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 300, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array(['E']),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}

        time_bins = np.arange(sim_dict['t_sim'])

        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        sptrains_bin_time = sana._SpikeAnalysis__time_binned_sptrains_X(
            X=X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        sptrains_bin_space_time = sana._time_and_space_binned_sptrains_X(
            X, positions, sptrains_bin_time,
            dtype=np.uint16)

        gt = np.zeros(((sana.space_bins.size - 1)**2, time_bins.size))
        gt[1, 1] = 1
        gt[1, 100] = 1
        gt[2, 200] = 1
        gt[4, 250] = 1

        self.assertTrue(np.all(sptrains_bin_space_time.toarray() == gt))

    def test_SpikeAnalysis_time_and_space_binned_sptrains_X_06(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 100
        spikes = np.array(list(zip([4, 1, 2, 1], [1.5, 200.5, 250.5, 100.5])),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one centered unit per spatial bin
        x, y = np.meshgrid(np.arange(-5, 5) + 0.5, np.arange(-5, 5) + 0.5)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 300, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array(['E']),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}

        time_bins = np.arange(sim_dict['t_sim'])

        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        sptrains_bin_time = sana._SpikeAnalysis__time_binned_sptrains_X(
            X=X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        sptrains_bin_space_time = sana._time_and_space_binned_sptrains_X(
            X, positions, sptrains_bin_time,
            dtype=np.uint16)

        gt = np.zeros(((sana.space_bins.size - 1)**2, time_bins.size))
        gt[4, 1] = 1
        gt[1, 200] = 1
        gt[2, 250] = 1
        gt[1, 100] = 1

        self.assertTrue(np.all(sptrains_bin_space_time.toarray() == gt))

    def test_SpikeAnalysis_extract_center_disc_1mm2_00(self):
        ''' test SpikeAnalysis.__extract_center_disc_1mm2()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        sim_dict, net_dict, ana_dict, spikes, positions = \
            self.dummy_definitions_for__extract_center_disc1mm2()
        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        spikes_1mm2, positions_1mm2 = \
            sana._extract_center_disc_1mm2(spikes, positions)

        bools = []
        for x, y in zip(positions_1mm2['x-position_mm'],
                        positions_1mm2['y-position_mm']):
            bools.append((x**2 + y**2) * np.pi <= 1)

        self.assertTrue(
            np.all(bools),
            'Remaining positions are not within center disc of 1mm2.')

    def test_SpikeAnalysis_extract_center_disc_1mm2_01(self):
        ''' test SpikeAnalysis.__extract_center_disc_1mm2()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        sim_dict, net_dict, ana_dict, spikes, positions = \
            self.dummy_definitions_for__extract_center_disc1mm2()
        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        spikes_1mm2, positions_1mm2 = \
            sana._extract_center_disc_1mm2(spikes, positions)

        spikes_1mm2_res = np.array([(4, 0.3),  # id was 6
                                    (0, 10.0),
                                    (1, 12.1),
                                    (4, 12.1),  # id was 6
                                    (3, 13.3),  # id was 4
                                    ], dtype=spikes.dtype)

        bools = []
        for i in np.arange(len(spikes_1mm2)):
            bools.append(np.all(spikes_1mm2[i] == spikes_1mm2_res[i]))

        self.assertTrue(np.all(bools), 'Spike data of 1mm2 incorrect.')

    def dummy_definitions_for__extract_center_disc1mm2(self):
        '''
        '''
        # dummy sim_dict, net_dict, and ana_dict
        sim_dict = {'t_presim': 0, 't_sim': 100, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array(['E']),
                    'num_neurons': np.array([5]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}

        # dtypes for spikes and positions
        dtype_spikes = {'names': ('nodeid', 'time_ms'),
                        'formats': ('i4', 'f8')}
        dtype_positions = {
            'names': (
                'nodeid',
                'x-position_mm',
                'y-position_mm'),
            'formats': (
                'i4',
                'f8',
                'f8')}

        # dummy spikes and positions
        spikes = np.array([(5, 0.1),
                           (6, 0.3),
                           (2, 0.5),
                           (2, 0.6),
                           (2, 1.4),
                           (0, 10.0),
                           (1, 12.1),
                           (6, 12.1),
                           (4, 13.3),
                           ], dtype=dtype_spikes)
        positions = np.array([(0, 0.1, 0.1),
                              (1, 0.05, 0.4),
                              (2, 1.5, 0.3),  # outside
                              (3, 0.2, 0.1),  # does not spike
                              (4, 0.3, 0.4),
                              (5, 0.5, 0.7),  # outside
                              (6, 0.3, 0.1),
                              ], dtype=dtype_positions)
        return sim_dict, net_dict, ana_dict, spikes, positions
