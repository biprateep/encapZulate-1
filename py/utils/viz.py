# @Author: Brett Andrews <andrews>
# @Date:   2019-05-28 14:05:49
# @Last modified by:   andrews
# @Last modified time: 2019-06-20 09:06:65

"""Visualization utilities."""

from pathlib import Path

from astropy.visualization import make_lupton_rgb
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from .fileio import construct_path_out, load_config
from .utils import import_model


class PlotModel(object):
    """Make plots.

    Args:
        run_name (str): Run name. Default is None.
        path (str): Path to results. Default is None.
    """

    def __init__(self, run_name=None, path=None):
        self.run_name, self.path_results = construct_path_out(run_name=run_name, path=path)
        self.path_logs = self.path_results / "logs"
        self.path_pasquet = Path(__file__).resolve().parents[3] / "data" / "pasquet2019"
        self._labels = None
        self._logs = None

    @property
    def labels(self):
        if self._labels is None:
            print("Loading labels...")
            self._labels = pd.read_hdf(self.path_pasquet / "labelsBinned.h5", "data")

        return self._labels

    @property
    def logs(self):
        if self._logs is None:
            self._logs = pd.read_csv(self.path_logs / "log.csv")
        return self._logs

    def _accuracy(self, ax):
        ax.plot(self.logs.epoch, self.logs.capsnet_acc, label="train")
        ax.plot(self.logs.epoch, self.logs.val_capsnet_acc, label="validation")
        ax.set_ylabel("accuracy")
        ax.legend()
        return ax

    def _loss(self, ax):
        ax.plot(self.logs.epoch, self.logs.capsnet_loss, label="margin")
        ax.plot(self.logs.epoch, self.logs.decoder_loss, label="decoder")
        ax.plot(self.logs.epoch, self.logs.loss, label="total")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        return ax

    def _to_ind(self, page, row, col):
        """Convert from (page, row, column) of galaxies.pdf to index of labelsMerged.csv."""
        page -= 1
        row -= 1
        col -= 1
        return page * 150 + row * 10 + col

    def _load_image_data(self, ind):
        obj = self.labels.iloc[ind]
        galid = f"{obj.plate:04}-{obj.mjd}-{obj.fiberID:03}"

        fin = (
            self.path_pasquet
            / "cubes"
            / f"{obj.plate:04}"
            / f"{obj.mjd}"
            / f"{galid}.npy"
        )
        return np.load(fin)

    def plot_logs(self, path_out=None, savefig=False, **kwargs):
        """Plot accuracy and loss for a model.

        Args:
            path_out (str): Output path. Default is None.
            savefig (bool): If True save figure to ``path_out``. Default is
                True.
            **kwargs: Keyword arguments to pass to ``plt.subplots()``.

        Returns:
            ``matplotlib.figure.Figure``
        """
        fig, axes = plt.subplots(nrows=2, **kwargs)
        axes[0] = self._accuracy(ax=axes[0])
        axes[1] = self._loss(ax=axes[1])

        path_out = path_out if path_out is not None else self.path_logs
        fout = path_out if path_out.is_file() else path_out / "logs.pdf"
        if savefig:
            fig.savefig(fout)

        return fig

    def plot_accuracy(self, path_out=None, savefig=False, **kwargs):
        """Plot accuracy for a model.

        Args:
            path_out (str): Output path. Default is None.
            savefig (bool): If True save figure to ``path_out``. Default is
                False.
            **kwargs: Keyword arguments to pass to ``plt.subplots()``.

        Returns:
            ``matplotlib.figure.Figure``
        """
        fig, ax = plt.subplots(**kwargs)
        ax = self._accuracy(ax=ax)

        path_out = path_out if path_out is not None else self.path_logs
        fout = path_out if path_out.is_file() else path_out / "accuracy.pdf"
        if savefig:
            fig.savefig(fout)

        return fig

    def plot_loss(self, path_out=None, savefig=False, **kwargs):
        """Plot loss for a model.

        Args:
            path_out (str): Output path. Default is None.
            savefig (bool): If True save figure to ``path_out``. Default is
                False.
            **kwargs: Keyword arguments to pass to ``plt.subplots()``.

        Returns:
            ``matplotlib.figure.Figure``
        """
        fig, ax = plt.subplots(**kwargs)
        ax = self._loss(ax=ax)

        path_out = path_out if path_out is not None else self.path_logs
        fout = path_out if path_out.is_file() else path_out / "loss.pdf"
        if savefig:
            fig.savefig(fout)

        return fig

    def plot_ugriz_gri(self, inds, path_out=None, savefig=False, rgb_kwargs=None, **kwargs):
        """Plot images for ugriz bands individually and gri composite.

        Args:
            inds (list): Indices of galaxies in ``labels`` or tuples of
                (page, row, column) from ``galaxies.pdf``.
            path_out (str): Output path. Default is None.
            savefig (bool): If True save figure to ``path_out``. Default is
                False.
            rgb_kwargs: Keyword arguments to pass to
                ``make_lupton_rgb()``. Default is None.
            **kwargs: Keyword arguments to pass to ``plt.subplots()``.

        Returns:
            ``matplotlib.figure.Figure``
        """
        inds = [self._to_ind(ind) if isinstance(ind, tuple) else ind for ind in inds]
        if rgb_kwargs is None:
            rgb_kwargs = {"stretch": 0.6, "Q": 5}

        nrows = len(inds)
        if "figsize" not in kwargs:
            kwargs["figsize"] = (12, nrows * 3)

        fig, axes = plt.subplots(nrows=nrows, ncols=6, **kwargs)
        axes = np.atleast_2d(axes)

        image_types = ["u", "g", "r", "i", "z", "gri"]
        for ax, title in zip(axes[0], image_types):
            ax.set_title(title)

        for ind, row in zip(inds, axes):
            data = self._load_image_data(ind=ind)
            u, g, r, i, z = np.transpose(data, axes=(2, 0, 1))

            for jj, band in enumerate((u, g, r, i, z)):
                row[jj].imshow(band, origin="lower", cmap="Greys_r")

            rgb = make_lupton_rgb(i, r, g, **rgb_kwargs)
            row[5].imshow(rgb, origin="lower")

            row[0].annotate(
                f"z = {self.labels.z.iloc[ind]:.4}",
                xy=(0.05, -0.15),
                xycoords="axes fraction",
            )

            for ax in row:
                ax.grid(False)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        path_out = path_out if path_out is not None else self.path_logs
        fout = path_out if path_out.is_file() else path_out / "ugriz_gri.pdf"
        if savefig:
            fig.savefig(fout)

        return fig

    def plot_gri_recon(
        self,
        inds,
        path_config=None,
        checkpoint=None,
        path_out=None,
        savefig=False,
        rgb_kwargs=None,
        **kwargs,
    ):
        """
        inds (list): Indices of galaxies in ``labels`` or tuples of
            (page, row, column) from ``galaxies.pdf``.
        path_config (str): Path to config file used to train model.
            Default is None.
        checkpoint (int): Checkpoint to load weights from. Default is
            None.
        """
        inds = [self._to_ind(ind) if isinstance(ind, tuple) else ind for ind in inds]
        if rgb_kwargs is None:
            rgb_kwargs = {"stretch": 0.6, "Q": 5}

        config = load_config(path_config, verbose=False)
        config.pop("checkpoint")

        # load model
        CapsNet = import_model(config["model_name"])
        __, eval_model = CapsNet(**config)

        nrows = len(inds)
        if "figsize" not in kwargs:
            kwargs["figsize"] = (6, nrows * 3)

        fig, axes = plt.subplots(nrows=nrows, ncols=2, **kwargs)
        axes = np.atleast_2d(axes)

        image_types = ["truth", "recon"]
        for ax, title in zip(axes[0], image_types):
            ax.set_title(title)

        for ind, row in zip(inds, axes):
            data = self._load_image_data(ind=ind)
            data = data[16:48, 16:48]  # TODO crop to inner 32x32

            from ..base.run_model import predict

            y_class, y_prob, recon = predict(
                model=eval_model,
                images=np.expand_dims(data, axis=0),
                checkpoint=checkpoint,
                **config
            )

            u, g, r, i, z = np.transpose(data, axes=(2, 0, 1))
            u_recon, g_recon, r_recon, i_recon, z_recon = np.transpose(recon[0], axes=(2, 0, 1))

            rgb = make_lupton_rgb(i, r, g, **rgb_kwargs)
            row[0].imshow(rgb, origin="lower")

            rgb_recon = make_lupton_rgb(i_recon, r_recon, g_recon, **rgb_kwargs)
            row[1].imshow(rgb_recon, origin="lower")

            z_classes = (self.labels.z_class.iloc[ind] // 18, y_class[0])  # TODO for 10 bins only!
            for ii, (ax, z_class) in enumerate(zip(row, z_classes)):

                if ii == 0:
                    redshift = self.labels.z.iloc[ind]
                else:
                    bin_edges = np.linspace(0, 0.4, 10)  # TODO for 10 bins only!
                    redshift = np.mean([bin_edges[z_class], bin_edges[z_class + 1]])

                ax.annotate(
                    # f"z = {redshift:.4}\n"
                    f"z_class = {z_class}",
                    xy=(0.05, -0.15),
                    xycoords="axes fraction",
                )

            for ax in row:
                ax.grid(False)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        path_out = path_out if path_out is not None else self.path_logs
        fout = path_out if path_out.is_file() else path_out / "gri_recon.pdf"
        if savefig:
            fig.savefig(fout)

        return fig
