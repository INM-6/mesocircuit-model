#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mesocircuit-project test suite.

"""

import numpy as np
import unittest
import mesocircuit.analysis.spike_analysis as spike_analysis
import mesocircuit.mesocircuit_framework as mesoframe
from mesocircuit.parameterization import helpers_network as helpnet


class TestSuite(unittest.TestCase):
    """
    test methods and modules
    """

    def test_TestSuite_00(self):
        '''test TestSuite'''
        self.assertTrue(True)

    def test_spike_analysis_time_binned_sptrains_X_00(self):
        '''test spike_analysis.__time_binned_sptrains_X()
        '''
        N_X = 1000
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', float)])
        time_bins = np.arange(1000.)

        sptrains_bin_time = spike_analysis._time_binned_sptrains_X(
            N_X=N_X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        self.assertTrue(np.all(sptrains_bin_time.toarray() == np.eye(N_X)))

    def test_spike_analysis_time_binned_sptrains_X_01(self):
        '''test spike_analysis._time_binned_sptrains_X()
        '''
        N_X = 50
        spikes = np.array(list(zip([1, 1, 2, 4], [1.5, 100.5, 200.5, 250.5])),
                          dtype=[('nodeid', int), ('time_ms', float)])
        time_bins = np.arange(1000.)

        sptrains_bin_time = spike_analysis._time_binned_sptrains_X(
            N_X=N_X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        gt = np.zeros((N_X, time_bins.size))
        gt[1, 1] = 1
        gt[1, 100] = 1
        gt[2, 200] = 1
        gt[4, 250] = 1

        self.assertTrue(np.all(sptrains_bin_time.toarray() == gt))

    def test_spike_analysis_time_binned_sptrains_X_02(self):
        '''test spike_analysis._time_binned_sptrains_X()
        '''
        N_X = 50
        spikes = np.array(list(zip([4, 2, 1, 1], [1.5, 100.5, 200.5, 250.5])),
                          dtype=[('nodeid', int), ('time_ms', float)])
        time_bins = np.arange(1000.)

        sptrains_bin_time = spike_analysis._time_binned_sptrains_X(
            N_X=N_X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        gt = np.zeros((N_X, time_bins.size))
        gt[4, 1] = 1
        gt[2, 100] = 1
        gt[1, 200] = 1
        gt[1, 250] = 1

        self.assertTrue(np.all(sptrains_bin_time.toarray() == gt))

    def test_spike_analysis_time_binned_sptrains_X_03(self):
        '''test spike_analysis._time_binned_sptrains_X()
        '''
        N_X = 50
        spikes = np.array(list(zip([1, 4, 2, 1], [100.5, 1.5, 250.5, 200.5])),
                          dtype=[('nodeid', int), ('time_ms', float)])
        time_bins = np.arange(1000.)

        sptrains_bin_time = spike_analysis._time_binned_sptrains_X(
            N_X=N_X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        gt = np.zeros((N_X, time_bins.size))
        gt[4, 1] = 1
        gt[1, 100] = 1
        gt[1, 200] = 1
        gt[2, 250] = 1

        self.assertTrue(np.all(sptrains_bin_time.toarray() == gt))

    def test_spike_analysis_time_and_space_binned_sptrains_X_00(self):
        '''test spike_analysis._time_and_space_binned_sptrains_X()
        '''

        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one centered unit per spatial bin
        x, y = np.meshgrid(np.arange(-5, 5) + 0.5, np.arange(-5, 5) + 0.5)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        time_bins = np.arange(100.)
        space_bins = np.linspace(-5., 5., int(10. / 1. + 1))

        sptrains_bin_time = spike_analysis._time_binned_sptrains_X(
            N_X=N_X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        sptrains_bin_space_time = spike_analysis._time_and_space_binned_sptrains_X(
            positions, sptrains_bin_time, space_bins,
            dtype=np.uint16)

        gt = np.eye((space_bins.size - 1)**2)

        self.assertTrue(np.all(sptrains_bin_space_time.toarray() == gt))

    def test_spike_analysis_time_and_space_binned_sptrains_X_01(self):
        '''test spike_analysis._time_and_space_binned_sptrains_X()
        '''
        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one unit per spatial bin located on left-hand edge
        # (we count intervals [LH, RH>)
        x, y = np.meshgrid(np.arange(-5, 5), np.arange(-5, 5))
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        time_bins = np.arange(100.)
        space_bins = np.linspace(-5., 5., int(10. / 1. + 1))

        sptrains_bin_time = spike_analysis._time_binned_sptrains_X(
            N_X=N_X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        sptrains_bin_space_time = spike_analysis._time_and_space_binned_sptrains_X(
            positions, sptrains_bin_time, space_bins,
            dtype=np.uint16)

        gt = np.eye((space_bins.size - 1)**2)

        self.assertTrue(np.all(sptrains_bin_space_time.toarray() == gt))

    def test_spike_analysis_time_and_space_binned_sptrains_X_02(self):
        '''test spike_analysis._time_and_space_binned_sptrains_X()
        '''
        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one unit per spatial bin located near right-hand edge
        # (we count intervals [LH, RH>)
        x, y = np.meshgrid(np.arange(-5, 5) + 0.9999,
                           np.arange(-5, 5) + 0.9999)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        time_bins = np.arange(100.)
        space_bins = np.linspace(-5., 5., int(10. / 1. + 1))

        sptrains_bin_time = spike_analysis._time_binned_sptrains_X(
            N_X=N_X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        sptrains_bin_space_time = spike_analysis._time_and_space_binned_sptrains_X(
            positions, sptrains_bin_time, space_bins,
            dtype=np.uint16)

        gt = np.eye((space_bins.size - 1)**2)

        self.assertTrue(np.all(sptrains_bin_space_time.toarray() == gt))

    def test_spike_analysis_time_and_space_binned_sptrains_X_03(self):
        '''test spike_analysis._time_and_space_binned_sptrains_X()
        '''

        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one unit per spatial bin located on right-hand edge
        # (we count intervals [LH, RH>)
        x, y = np.meshgrid(np.arange(-5, 5) + 1, np.arange(-5, 5) + 1)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        time_bins = np.arange(1000.)
        space_bins = np.linspace(-2., 2., int(4. / 0.5 + 1))

        sptrains_bin_time = spike_analysis._time_binned_sptrains_X(
            N_X=N_X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        try:
            sptrains_bin_space_time = spike_analysis._time_and_space_binned_sptrains_X(
                positions, sptrains_bin_time, space_bins,
                dtype=np.uint16)
        except NotImplementedError:
            pass

    def test_spike_analysis_time_and_space_binned_sptrains_X_04(self):
        '''test spike_analysis._time_and_space_binned_sptrains_X()
        '''
        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one unit per spatial bin located on right-hand edge
        # (we count intervals [LH, RH>)
        x, y = np.meshgrid(np.arange(-5, 5) - 0.0001,
                           np.arange(-5, 5) - 0.0001)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        time_bins = np.arange(1000.)
        space_bins = np.linspace(-2., 2., int(4. / 0.5 + 1))

        sptrains_bin_time = spike_analysis._time_binned_sptrains_X(
            N_X=N_X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        try:
            sptrains_bin_space_time = spike_analysis._time_and_space_binned_sptrains_X(
                positions, sptrains_bin_time, space_bins,
                dtype=np.uint16)
        except NotImplementedError:
            pass

    def test_spike_analysis_time_and_space_binned_sptrains_X_05(self):
        '''test spike_analysis._time_and_space_binned_sptrains_X()
        '''
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
        space_bins = np.linspace(-5., 5., int(10. / 1. + 1))

        sptrains_bin_time = spike_analysis._time_binned_sptrains_X(
            N_X=N_X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        sptrains_bin_space_time = spike_analysis._time_and_space_binned_sptrains_X(
            positions, sptrains_bin_time, space_bins,
            dtype=np.uint16)

        gt = np.c_[np.eye((space_bins.size - 1)**2),
                   np.eye((space_bins.size - 1)**2),
                   np.eye((space_bins.size - 1)**2)]

        self.assertTrue(np.all(sptrains_bin_space_time.toarray() == gt))

    def test_spike_analysis_time_and_space_binned_sptrains_X_06(self):
        '''test spike_analysis._time_and_space_binned_sptrains_X()
        '''

        X = 'E'
        N_X = 100
        spikes = np.array(list(zip([1, 1, 2, 4], [1.5, 100.5, 200.5, 250.5])),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one centered unit per spatial bin
        x, y = np.meshgrid(np.arange(-5, 5) + 0.5, np.arange(-5, 5) + 0.5)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        time_bins = np.arange(1000.)
        space_bins = np.linspace(-5., 5., int(10. / 1. + 1))

        sptrains_bin_time = spike_analysis._time_binned_sptrains_X(
            N_X=N_X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        sptrains_bin_space_time = spike_analysis._time_and_space_binned_sptrains_X(
            positions, sptrains_bin_time,
            dtype=np.uint16)

        gt = np.zeros(((space_bins.size - 1)**2, time_bins.size))
        gt[1, 1] = 1
        gt[1, 100] = 1
        gt[2, 200] = 1
        gt[4, 250] = 1

        self.assertTrue(np.all(sptrains_bin_space_time.toarray() == gt))

    def test_spike_analysis_time_and_space_binned_sptrains_X_06(self):
        '''test spike_analysis._time_and_space_binned_sptrains_X()
        '''

        N_X = 100
        spikes = np.array(list(zip([4, 1, 2, 1], [1.5, 200.5, 250.5, 100.5])),
                          dtype=[('nodeid', int), ('time_ms', float)])

        # one centered unit per spatial bin
        x, y = np.meshgrid(np.arange(-5, 5) + 0.5, np.arange(-5, 5) + 0.5)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        time_bins = np.arange(1000.)
        space_bins = np.linspace(-5., 5., int(10. / 1. + 1))

        sptrains_bin_time = spike_analysis._time_binned_sptrains_X(
            N_X=N_X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        sptrains_bin_space_time = spike_analysis._time_and_space_binned_sptrains_X(
            positions, sptrains_bin_time, space_bins,
            dtype=np.uint16)

        gt = np.zeros(((space_bins.size - 1)**2, time_bins.size))
        gt[4, 1] = 1
        gt[1, 200] = 1
        gt[2, 250] = 1
        gt[1, 100] = 1

        self.assertTrue(np.all(sptrains_bin_space_time.toarray() == gt))

    def test_spike_analysis_extract_center_disc_1mm2_00(self):
        ''' test spike_analysis.__extract_center_disc_1mm2()
        '''
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

        spikes_1mm2, positions_1mm2 = \
            spike_analysis._extract_center_disc_1mm2(spikes, positions)

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

        spikes_1mm2, positions_1mm2 = \
            spike_analysis._extract_center_disc_1mm2(spikes, positions)

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

    def test_get_delay_lin_effective(self):
        '''
        Test effective mean delay from delays with linear distance dependency.
        '''

        radius = 1./np.sqrt(np.pi)
        beta = np.array([[0.232, 0.161], [0.125, 0.120]])
        delay_offset_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        prop_speed_matrix = np.array([[0.3, 0.3], [0.3, 0.3]])

        # function to be tested
        delay_lin_eff_mean_matrix, delay_lin_eff_std_matrix = \
            helpnet.get_delay_lin_effective(
                radius,
                beta,
                delay_offset_matrix,
                prop_speed_matrix)

        # alternative approach: sample mean distance and calculate delay
        def mean_distance_sampled(radius, beta, num_nodes, num_distances):
            xs = []
            ys = []
            for n in np.arange(num_nodes):
                accept = False
                while not accept:
                    x, y = 2 * radius * np.random.rand(2) - radius
                    dist_to_center = np.sqrt(x**2 + y**2)
                    if dist_to_center < radius:
                        accept = True
                        xs.append(x)
                        ys.append(y)

            distances = []
            for n in np.arange(num_distances):
                accept = False
                while not accept:
                    s, t = np.random.randint(low=0, high=num_nodes, size=2)
                    distance = np.sqrt((xs[t] - xs[s])**2 + (ys[t] - ys[s])**2)
                    if np.random.random() < np.exp(-distance / beta):
                        accept = True
                        distances.append(distance)
            mean_distance = np.mean(distances)
            return xs, ys, mean_distance

        num_nodes = 10000
        num_distances = 1000
        for i in np.arange(2):
            for j in np.arange(2):
                xs, ys, mean_distance = mean_distance_sampled(
                    radius, beta[i, j], num_nodes, num_distances)
                mean_delay_sampled = delay_offset_matrix[i, j] \
                    + mean_distance / prop_speed_matrix[i, j]

                # print(delay_lin_eff_mean_matrix[i, j], mean_delay_sampled)
                self.assertAlmostEqual(delay_lin_eff_mean_matrix[i, j] /
                                       mean_delay_sampled,
                                       1.,
                                       places=1)  # one decimal place (10%)
