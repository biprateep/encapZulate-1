# @Author: Brett Andrews <andrews>
# @Date:   2019-06-18 09:06:77
# @Last modified by:   andrews
# @Last modified time: 2019-06-19 10:06:25

"""Data augmentation utilities."""


def crop_center(shape_in, shape_out):
    """Crop input images to desired output shape.

    Args:
        shape_in (tuple): Input shape.
        shape_out (tuple): Output shape.

    Returns:
        tuple: indices to slice image array.
    """

    slices = []
    for length, delta in zip(shape_in, shape_out):
        assert isinstance(length, int), "shape_in must be a tuple of ints."
        assert isinstance(delta, int), "shape_out must be a tuple of ints."
        assert length >= delta, "Cropped shape cannot be larger than input shape."
        assert (length - delta) % 2 == 0, \
            "Cropped shape cannot be centered given input and output shapes."

        start = length // 2 - delta // 2
        slices.append((start, start + delta))

    return tuple(slices)


def consolidate_bins(labels, n_bins_in, n_bins_out):
    """Consolidate bins.

    Args:
        labels (array): Input labels.
        n_bins_in (int): Number of bins for input data.
        n_bins_out (int): Number of desired output bins.

    Returns:
        array: Labels consolidated into the desired number of output
            bins.
    """
    assert n_bins_in % n_bins_out == 0, \
        (f"Must choose a number of output classes ({n_bins_out}) divisible by the"
         f"initial number of output classes ({n_bins_in}).")

    bin_consolidation_factor = n_bins_in / n_bins_out
    return labels // bin_consolidation_factor
