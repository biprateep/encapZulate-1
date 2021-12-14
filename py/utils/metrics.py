"""Performance metrics for photo-z prediction."""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import mpl_scatter_density
import numpy as np
import scipy
from scipy.special import softmax
from scipy.stats import gaussian_kde


params = {
    "legend.fontsize": "x-large",
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
    "figure.facecolor": "w",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
}
plt.rcParams.update(params)


def bins_to_redshifts(ind_bin, z_min=None, num_class=None, dz=None, return_edges=False):
    """Convert bin labels into redshift bin edges or central values.

    Args:
        ind_bin (array): Index of redshift bins
        z_min (float): Minimum of the redshift range
        num_class (int): Number of redshift classes
        dz (float): Width of the redshift bins
        return_edges (bool): If True, return the bin edges, otherwise
            return the bin centers. Default is False.

    Returns:
        array: edges or centers or redshift bins (see `return_edges`
            arg)
    """
    ind_bin = np.atleast_1d(ind_bin)
    z_max = z_min + num_class * dz
    edges = np.arange(z_min, z_max + dz, dz)
    bin_edges = np.array([edges[:-1], edges[1:]]).T

    if return_edges:
        return bin_edges[ind_bin]
    else:
        return np.mean(bin_edges, axis=1)[ind_bin]


def probs_to_redshifts(
    prob, z_min=None, num_class=None, dz=None, conversion="max", **kwargs
):
    """Convert class probabilities from Neural net to redshift predictions

    Args:
        prob (array): array of all the class probabilities
        num_class (int): Number of prediction classes
        z_min (float): Minimum of the redshift range
        dz (float): Width of the redshift bins
        conversion (string): "weighted" or "max", take the modal prediction or probability weighted
    Returns:
        Predicted photo-Zs for the batch
    """
    assert conversion in [
        "weighted",
        "max",
    ], 'Conversion mode should be either "weighted" or "max" '
    np.save("prob", prob)
    if conversion == "max":
        redshifts = bins_to_redshifts(
            np.argmax(prob, axis=-1), z_min, num_class, dz, return_edges=False
        )

    if conversion == "weighted":
        classes = np.arange(num_class)
        redshifts = bins_to_redshifts(classes, z_min, num_class, dz, return_edges=False)
        redshifts = np.sum(prob * redshifts, axis=-1)

    return redshifts


def hodges_lehmann(data, max_pairs=1e6, random_seed=200):
    """The Hodges-Lehmann estimator.

    Adapted from code written by Rongpu Zhou.

    Args:
        data (1D array): Data set for which the estimator is being
            computed.
        max_pairs (int): If number of pairs is larger than this,
            randomly sample pairs.
        random_seed (int): Seed for randomly sampling pairs.

    Returns:
        float: H-L estimate
    """

    import itertools

    max_pairs = int(max_pairs)
    n_data = len(data)
    n_pairs = n_data * (n_data - 1) / 2

    if n_data == 0:
        raise ValueError("Must pass in non-empty array.")

    if n_pairs <= max_pairs:
        # non-identical indices
        ind1, ind2 = np.array(list(itertools.combinations(np.arange(n_data), 2))).T
        pair_means = np.mean([data[ind1], data[ind2]], axis=0)

        #  identical indices
        pair_means = np.concatenate([pair_means, data])

    else:
        if random_seed is not None:
            np.random.seed(random_seed)

        ind1, ind2 = np.random.choice(n_data, size=(max_pairs, 2)).transpose()
        pair_means = np.mean([data[ind1], data[ind2]], axis=0)

    return np.median(pair_means)


def temperature_scaling(temperature, logits):
    """Apply temperature scaling to calibrate PDFs of neural network.

    See S4.2 of https://arxiv.org/pdf/1706.04599.pdf

    Args:
        logits (array): Output of neural network.
        temperature (float): Constant used to scale logits to create PDFs.

    Returns:
        array: Output PDFs.
    """
    return softmax(logits / temperature, axis=1)


def best_temperature(logits, truth_hist):
    # implement strategy to find best temperature
    # minimize difference between N(z) and predicted N(z)?
    # for each instance minimize CRPS or PIT?

    temperature_best = scipy.optimize.minimize(mse, 0.5, args=(logits, truth_hist)).x[0]

    return temperature_best


# TODO Rename
def mse(temperature, logits, truth_hist):
    y_prob = temperature_scaling(temperature, logits)
    return np.sum((y_prob.sum(axis=0) - truth_hist) ** 2)


class Metrics(object):
    """Produce metrics for the model.

    Args:
        z_phot (array): Predicted photometric redshifts.
        z_spec (array): Measured spectroscopic redshifts.
    """

    def __init__(
        self, z_phot, z_spec, z_min=None, z_max=None, outlier_threshold=None, **kwargs
    ):
        z_mask = (z_spec >= z_min) & (z_spec <= z_max)
        self.z_phot = z_phot[z_mask]
        self.z_spec = z_spec[z_mask]
        self.z_min = z_min
        self.z_max = z_max
        self.outlier_threshold = outlier_threshold

        # TODO make convenience functions for all
        # Normalized residuals as defined in Cohen et al. (2000) Section 3
        self.delta_z_norm = (self.z_phot - self.z_spec) / (1 + self.z_spec)
        # Normalized median absolute deviation
        self.sigma_nmad = 1.4826 * np.median(
            np.abs(self.delta_z_norm - np.median(self.delta_z_norm))
        )
        self.bias = np.mean(self.delta_z_norm)
        # Number of objects larger than outlier threshold
        self.n_outlier = np.sum(np.abs(self.delta_z_norm) > self.outlier_threshold)

        # Outlier percentage
        self.percent_outlier = self.n_outlier * 100.0 / len(self.z_spec)

    def _gaussian(self, x, mean=0, sigma=1):
        return np.exp((-0.5 * ((x - mean) / sigma) ** 2)) / np.sqrt(2 * np.pi) / sigma

    def phot_vs_spec(self, show=False, ax=None, fig=None, **kwargs):
        """Photo-z vs. spec-z."""

        if ax is None:
            fig, ax = plt.subplots(
                subplot_kw={"projection": "scatter_density"},
                **kwargs,
            )

        print(f"Normalized MAD: {self.sigma_nmad:.6f}")
        print(f"{self.outlier_threshold:.2f} outliers: {self.percent_outlier:.6f}%")

        x = np.linspace(self.z_min, self.z_max, 10)
        outlier_upper = x + self.outlier_threshold * (1 + x)
        outlier_lower = x - self.outlier_threshold * (1 + x)
        ax.plot(x, outlier_upper, "k--")
        ax.plot(x, outlier_lower, "k--")

        # Define new cmap
        viridis = cm.get_cmap("viridis", 256)
        newcolors = viridis(np.linspace(0, 1, 256))
        white = np.array([1, 1, 1, 1])
        newcolors[:1, :] = white
        viridis_white = ListedColormap(newcolors, name="viridis_white")

        # Plot scatter density
        scatter_density = ax.scatter_density(
            self.z_spec, self.z_phot, cmap=viridis_white, dpi=50, downres_factor=1
        )
        cbar = fig.colorbar(
            scatter_density,
            fraction=0.046,
            pad=0.04,
        )
        #         cbar.ax.tick_params(labelsize=40)
        cbar.set_label(label="Number of galaxies per pixel", fontsize=20)

        ax.plot(x, x, linewidth=1.5, color="grey")

        ax.set_xlim([self.z_min, self.z_max])
        ax.set_ylim([self.z_min, self.z_max])
        ax.set_xlabel(r"$z_{\mathrm{spec}}$", fontsize=40)
        ax.set_ylabel(r"$z_{\mathrm{phot}}$", fontsize=40)
        ax.yaxis.grid(alpha=0.8, ls="--")
        ax.xaxis.grid(alpha=0.8)
        ax.set_aspect("equal")
        xticks = ax.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)

        textstr = "\n".join(
            (
                r"$\sigma_{\mathrm{NMAD}}=%.5f$" % (self.sigma_nmad,),
                r"$\mathrm{f}_{\mathrm{outlier}}=%.2f$" % (self.percent_outlier,),
                r"$\langle \frac{\Delta z}{1+z_{\mathrm{spec}}} \rangle=%.5f$"
                % (self.bias),
            )
        )

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round, pad=0.7", facecolor="none", alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(
            0.05,
            0.97,
            textstr,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment="top",
            bbox=props,
        )

        if show:
            plt.show()

        return ax

    def means(self, show=False, ax=None, num_z_bins=8, **kwargs):
        """NMAD and Mean deviation"""

        z_bins = np.linspace(self.z_min, self.z_max, num_z_bins + 1)
        z_binsize = z_bins[1] - z_bins[0]
        dz_mean = np.zeros_like(z_bins)
        z_bins_mean = np.zeros_like(z_bins)
        sigma_nmad_bins = np.zeros_like(z_bins)

        for ii, z_bin in enumerate(z_bins[:-1]):
            mask1 = (self.z_spec >= z_bin) & (self.z_spec < (z_bin + z_binsize))
            dz_mean[ii] = hodges_lehmann(self.delta_z_norm[mask1], max_pairs=1e6)
            z_bins_mean[ii] = hodges_lehmann(self.z_spec[mask1], max_pairs=1e6)
            sigma_nmad_bins[ii] = 1.4826 * np.median(
                (np.abs(self.delta_z_norm[mask1] - np.median(self.delta_z_norm[mask1])))
            )

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "scatter_density"})

        ax.scatter_density(
            self.z_spec, self.delta_z_norm, cmap="Greys", dpi=72, downres_factor=1
        )
        ax.plot(z_bins_mean, dz_mean, "r", linewidth=1.5, label="Mean deviation")
        ax.plot(
            z_bins_mean,
            sigma_nmad_bins,
            color="orange",
            ls="--",
            linewidth=1.5,
            label=r"$\sigma_{\mathrm{NMAD}}$",
        )

        # plot lines of constant z_phot
        z_phot_fixed = np.linspace(self.z_min, self.z_max, 5)
        x = np.linspace(self.z_min, self.z_max, 100)
        for it in z_phot_fixed:
            ax.plot(x, (it - x) / (1 + x), "--", color="gray", lw=1, alpha=0.5)

        ax.set_ylabel(r"$\dfrac{\Delta z}{1 + z_\mathrm{spec}}$", fontsize=30)
        ax.set_xlabel("$z_{spec}$", fontsize=20)
        ax.axis([self.z_min, self.z_max, -0.1, 0.1])
        ax.axhline(0, linestyle="--", color="black")
        ax.set_ylim(-0.02, 0.02)
        ax.grid()
        ax.legend(loc="lower left")

        if show:
            plt.show()

        return ax

    def dz_norm(self, show=False, ax=None, **kwargs):
        """Histogram of normalized redshift residuals."""
        print("Bias: {:.6f}".format(self.bias))
        print("Sigma MAD: {:.6f}".format(self.sigma_nmad))

        if ax is None:
            fig, ax = plt.subplots(**kwargs)

        pop, bins, patches = ax.hist(
            self.delta_z_norm,
            bins=100,
            histtype="stepfilled",
            color="gray",
            alpha=0.5,
            density=True,
        )

        bin_width = bins[1] - bins[0]
        x = np.linspace(bins.min(), bins.max(), 201)
        ax.plot(
            x,
            self._gaussian(x, self.bias, self.sigma_nmad) * pop.sum() * bin_width,
            c="C0",
            ls="--",
        )
        ax.grid(alpha=0.5)
        ax.set_ylabel("Relative Frequency", fontsize=20)
        ax.set_xlabel(r"$\dfrac{\Delta z}{1 + z_\mathrm{spec}}$", fontsize=30)
        ax.set_xlim([-1 * self.outlier_threshold, self.outlier_threshold])

        if show:
            plt.show()

        return ax

    def full_diagnostic(self, figsize=(26, 13), show=False):
        fig = plt.figure(figsize=figsize)
        ax0 = plt.subplot2grid((2, 2), (0, 0), rowspan=2, projection="scatter_density")
        ax1 = plt.subplot2grid((2, 2), (0, 1), projection="scatter_density")
        ax2 = plt.subplot2grid((2, 2), (1, 1))
        ax0 = self.phot_vs_spec(ax=ax0, fig=fig)
        ax1 = self.means(ax=ax1)
        ax2 = self.dz_norm(ax=ax2)

        if show:
            plt.show()

        return ax0, ax1, ax2