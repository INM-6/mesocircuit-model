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
                          dtype=[('nodeid', int), ('time_ms', object)])

        time_bins = np.arange(1000)

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 1000, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array(['E']),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}


        sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

        sptrains_bin_time = sana._SpikeAnalysis__time_binned_sptrains_X(
            X=X,
            spikes=spikes,
            time_bins=time_bins, dtype=int)

        self.assertTrue(np.all(sptrains_bin_time.todense() == np.eye(N_X)))


    def test_SpikeAnalysis_time_and_space_binned_sptrains_X_00(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', object)])

        # one centered unit per spatial bin
        x, y = np.meshgrid(np.arange(-5, 5) + 0.5, np.arange(-5, 5) + 0.5)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        # one spike per neuron per spatial bin
        time_bins = np.arange(100)

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 100, 'sim_resolution': 0.1}
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

        gt = np.eye(N_X)

        self.assertTrue(np.all(sptrains_bin_space_time.todense() == gt))


    def test_SpikeAnalysis_time_and_space_binned_sptrains_X_01(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', object)])

        # one unit per spatial bin located on left-hand edge
        # (we count intervals [LH, RH>)
        x, y = np.meshgrid(np.arange(-5, 5), np.arange(-5, 5))
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        # one spike per neuron per spatial bin
        time_bins = np.arange(100)

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 100, 'sim_resolution': 0.1}
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

        gt = np.eye(N_X)

        self.assertTrue(np.all(sptrains_bin_space_time.todense() == gt))

    def test_SpikeAnalysis_time_and_space_binned_sptrains_X_02(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', object)])

        # one unit per spatial bin located near right-hand edge
        # (we count intervals [LH, RH>)
        x, y = np.meshgrid(np.arange(-5, 5) + 0.9999, np.arange(-5, 5) + 0.9999)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        # one spike per neuron per spatial bin
        time_bins = np.arange(100)

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 100, 'sim_resolution': 0.1}
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

        gt = np.eye(N_X)

        self.assertTrue(np.all(sptrains_bin_space_time.todense() == gt))

    def test_SpikeAnalysis_time_and_space_binned_sptrains_X_03(self):
        '''test SpikeAnalysis.__time_binned_sptrains_X()
        '''
        from core.analysis.spike_analysis import SpikeAnalysis

        X = 'E'
        N_X = 100
        spikes = np.array(list(zip(np.arange(N_X), np.arange(N_X) + 0.5)),
                          dtype=[('nodeid', int), ('time_ms', object)])

        # one unit per spatial bin located on right-hand edge
        # (we count intervals [LH, RH>)
        x, y = np.meshgrid(np.arange(-5, 5) + 1, np.arange(-5, 5) + 1)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        # one spike per neuron per spatial bin
        time_bins = np.arange(100)

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 100, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array(['E']),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}


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
                          dtype=[('nodeid', int), ('time_ms', object)])

        # one unit per spatial bin located on right-hand edge
        # (we count intervals [LH, RH>)
        x, y = np.meshgrid(np.arange(-5, 5) - 0.0001, np.arange(-5, 5) - 0.0001)
        positions = {'x-position_mm': x.ravel(),
                     'y-position_mm': y.ravel()}

        # one spike per neuron per spatial bin
        time_bins = np.arange(100)

        # dummy simulation dicts allowing creating SpikeAnalysis instance
        sim_dict = {'t_presim': 0, 't_sim': 100, 'sim_resolution': 0.1}
        net_dict = {'populations': np.array(['E']),
                    'num_neurons': np.array([N_X]),
                    'extent': 10}
        ana_dict = {'binsize_time': 1, 't_transient': 0, 'binsize_space': 1}


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
