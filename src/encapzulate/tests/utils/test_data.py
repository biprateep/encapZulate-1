# @Author: Brett Andrews <andrews>
# @Date:   2019-06-18 09:06:69
# @Last modified by:   andrews
# @Last modified time: 2019-06-19 10:06:45

import numpy as np
import pytest

from encapzulate.utils.data import consolidate_bins, crop_center


class TestData(object):

    @pytest.mark.parametrize(
        "shape_in, shape_out, indices",
        [
            ((3, 3), (3, 3), ((0, 3), (0, 3))),
            ((3, 3), (1, 1), ((1, 2), (1, 2))),
            ((5, 5), (3, 3), ((1, 4), (1, 4))),
            ((8, 8), (4, 4), ((2, 6), (2, 6))),
            ((10, 10), (8, 8), ((1, 9), (1, 9))),
            ((10, 8), (8, 8), ((1, 9), (0, 8))),
            ((10, 7), (6, 5), ((2, 8), (1, 6))),
        ]
    )
    def test_crop_center(self, shape_in, shape_out, indices):
        assert crop_center(shape_in, shape_out) == indices

    @pytest.mark.parametrize(
        "shape_in, shape_out",
        [
            ((5.1, 5), (3, 3)),
            ((8, 8.0), (4, 4)),
            ((10, 10), (8.1, 8)),
            ((10, 10), (8, 8.0)),
        ]
    )
    def test_crop_center_assert_int_shapes(self, shape_in, shape_out):
        with pytest.raises(AssertionError) as ee:
            crop_center(shape_in, shape_out)

        assert "must be a tuple of ints." in str(ee.value)

    @pytest.mark.parametrize(
        "shape_in, shape_out",
        [
            ((5, 5), (2, 2)),
            ((8, 8), (3, 3)),
            ((10, 10), (5, 5)),
        ]
    )
    def test_crop_center_not_centered(self, shape_in, shape_out):
        with pytest.raises(AssertionError) as ee:
            crop_center(shape_in, shape_out)

        assert "Cropped shape cannot be centered given input and output shapes." in str(ee.value)

    @pytest.mark.parametrize(
        "shape_in, shape_out",
        [
            ((5, 5), (7, 7)),
            ((5, 5), (8, 8)),
            ((8, 8), (9, 9)),
            ((8, 8), (10, 10)),
        ]
    )
    def test_crop_center_crop_too_big(self, shape_in, shape_out):
        with pytest.raises(AssertionError) as ee:
            crop_center(shape_in, shape_out)

        assert "Cropped shape cannot be larger than input shape." in str(ee.value)

    @pytest.mark.parametrize(
        "n_bins_out, labels_out",
        [
            (1, np.zeros(6)),
            (2, np.array([0, 0, 0, 1, 1, 0])),
            (3, np.array([0, 0, 0, 2, 2, 0])),
            (4, np.array([0, 0, 0, 3, 2, 0])),
            (6, np.array([0, 0, 1, 5, 4, 1])),
            (12, np.array([0, 1, 2, 11, 8, 2])),
        ]
    )
    def test_consolidate_bins(self, n_bins_out, labels_out):
        labels = np.array([0, 1, 2, 11, 8, 2])  # 12 possible input labels
        assert (consolidate_bins(labels, n_bins_in=12, n_bins_out=n_bins_out) == labels_out).all()

    @pytest.mark.parametrize("n_bins_out", [5, 7, 8, 9, 10, 11])
    def test_consolidate_bins_not_divisible(self, n_bins_out):
        n_bins_in = 12
        labels = np.array([0, 1, 2, 11, 8, 2])
        with pytest.raises(AssertionError) as ee:
            consolidate_bins(labels, n_bins_in=n_bins_in, n_bins_out=n_bins_out)

        assert (f"Must choose a number of output classes ({n_bins_out}) divisible by the"
                f"initial number of output classes ({n_bins_in}).") in str(ee)
