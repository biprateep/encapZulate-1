# @Author: Brett Andrews <andrews>
# @Date:   2019-08-07 10:08:76
# @Last modified by:   andrews
# @Last modified time: 2019-08-07 14:08:53

import numpy as np
import pytest

from encapzulate.utils.metrics import bins_to_redshifts, hodges_lehmann


ind_bin = np.arange(10)
bin_edges_all = np.arange(0, 0.101, 0.01)
bin_edges = np.array([bin_edges_all[:-1], bin_edges_all[1:]]).T
bin_centers = np.mean(bin_edges, axis=1)

data_hl = np.array([1.1, 2.3, 0.7, -4, 10])


class TestMetrics(object):

    @pytest.mark.parametrize(
        "ind_bin, z_min, num_class, dz, edges",
        [
            (ind_bin, 0, 10, 0.01, bin_edges),
            (0, 0, 10, 0.01, np.array([0, 0.01])),
            (2, 0, 10, 0.01, np.array([0.02, 0.03])),
            (9, 0, 10, 0.01, np.array([0.09, 0.1])),
        ]
    )
    def test_bins_to_redshifts_edges(self, ind_bin, z_min, num_class, dz, edges):
        assert (bins_to_redshifts(ind_bin, z_min, num_class, dz, return_edges=True) == edges).all()

    @pytest.mark.parametrize(
        "ind_bin, z_min, num_class, dz, centers",
        [
            (ind_bin, 0, 10, 0.01, bin_centers),
            (0, 0, 10, 0.01, 0.005),
            (2, 0, 10, 0.01, 0.025),
            (9, 0, 10, 0.01, 0.095),
        ]
    )
    def test_bins_to_redshifts_centers(self, ind_bin, z_min, num_class, dz, centers):
        assert (bins_to_redshifts(ind_bin, z_min, num_class, dz, return_edges=False) == centers).all()

    @pytest.mark.parametrize("data", [data_hl])
    def test_hodges_lehmann_all_pairs(self, data):
        assert hodges_lehmann(data) == 1.5

    @pytest.mark.parametrize("data", [data_hl])
    def test_hodges_lehmann_random(self, data):
        assert hodges_lehmann(data, max_pairs=3) == 5.35

    def test_hodges_lehmann_no_data(self):
        with pytest.raises(ValueError) as ee:
            hodges_lehmann(np.array([]))

        assert "Must pass in non-empty array." in str(ee.value)
