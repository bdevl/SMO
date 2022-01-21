import SMO.factories
import numpy as np
import torch
from genutils import ensure_folder, MatplotlibVideo
import matplotlib.pyplot as plt
from lamp.utils import reparametrize
from tqdm import tqdm
import matplotlib.patches as patches
from scipy.stats import kde
from SMO.optimization.objectives import WrapperModel
from typing import Union, Literal, List, Type, final
from matplotlib import colors
import tikzplotlib

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 11,
        "text.latex.preamble": [r"\usepackage{bm}"],
    }
)


def savefig_tikz(fig, plotdir, name, uuid=None, close_fig=True):

    if uuid is not None:
        ensure_folder(plotdir + uuid)
        savename = plotdir + uuid + "/" + name
    else:
        plotdir = ensure_folder(plotdir)
        savename = plotdir + name

    tikzplotlib.save(savename + ".tex", axis_height="\\figH", axis_width="\\figW")

    plt.savefig(savename + ".png", dpi=150)

    if close_fig:
        plt.close(fig)


def savefig(fig, plotdir, name, uuid, close_fig):

    ensure_folder(plotdir + uuid)

    plt.savefig(
        plotdir + uuid + "/" + name + ".png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0.08,
    )

    plt.savefig(
        plotdir + uuid + "/" + name + ".pdf",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0.08,
    )

    if close_fig:
        plt.close(fig)


class HmgWrappedSampler(object):
    def __init__(self, factory):

        self._factory = factory
        self._rf, _ = factory.rf()
        self._hmg = factory.hmg()
        self._cutoff = factory.cutoff
        self._htransform = factory.htransform()

    def set_phi(self, phi):
        self._rf.set_phi(phi)

    def sample(self, N, return_Xg=False):

        assert N > 1

        with torch.no_grad():
            Xg = self._rf.rsample_batch(N)
            kappas = self._hmg.homogenize_img(self._htransform(Xg), AcknowledgeRaw=True)
            kappas = torch.tensor(
                [[kappa[t] for t in self._factory.target] for kappa in kappas],
                dtype=Xg.dtype,
                device=Xg.device,
            )

            X_phases = (Xg > self._cutoff).detach().cpu().numpy().astype(np.float)

        if not return_Xg:
            return X_phases, kappas.detach().cpu().numpy()
        else:
            return Xg, kappas


def compare_microstructures_splitvf(
    factories,
    phi_init,
    phi_conv,
    N_examples=4,
    figsize=(9.5, 4.5),
    colorbar=False,
    final_iteration=None,
    cmap=None,
    labels=None,
    fontsize_xlabel=None,
):

    fig, axi = plt.subplots(2, N_examples, figsize=figsize)

    assert N_examples in [4, 6]

    sampler_A = HmgWrappedSampler(factories[0])
    sampler_A.set_phi(phi_init[0])
    X_init_A, kappas_init_A = sampler_A.sample(int(N_examples / 2))

    sampler_B = HmgWrappedSampler(factories[1])
    sampler_B.set_phi(phi_init[1])
    X_init_B, kappas_init_B = sampler_B.sample(int(N_examples / 2))

    for n in range(N_examples):

        plt.sca(axi[0, n])

        if N_examples == 6 and n == 1:
            plt.title(r"(a) volume fraction : 0.5")
        elif N_examples == 6 and n == 4:
            plt.title(r"(b) volume fraction : 0.3")

        if n <= (int(N_examples / 2) - 1):
            x = X_init_A[n].squeeze()
            kappa1 = kappas_init_A[n, 0].item()
            kappa2 = kappas_init_A[n, 1].item()
        else:
            x = X_init_B[n % 2].squeeze()
            kappa1 = kappas_init_B[n % 2, 0].item()
            kappa2 = kappas_init_B[n % 2, 1].item()

        imshow_grayscale(x, cmap=cmap)
        plt.gca().set_xticks([], [])
        plt.gca().set_yticks([], [])
        xlabel = (
            r"$\kappa_1 = {:.1f}$".format(kappa1)
            + ""
            + " $\kappa_2 = {:.1f}$".format(kappa2)
        )
        plt.xlabel(xlabel, fontsize=fontsize_xlabel)

    sampler_A.set_phi(phi_conv[0])
    sampler_B.set_phi(phi_conv[1])

    X_conv_A, kappas_conv_A = sampler_A.sample(int(N_examples / 2))
    X_conv_B, kappas_conv_B = sampler_B.sample(int(N_examples / 2))

    for n in range(N_examples):
        plt.sca(axi[1, n])

        if n <= (int(N_examples / 2) - 1):
            x = X_conv_A[n].squeeze()
            kappa1 = kappas_conv_A[n, 0].item()
            kappa2 = kappas_conv_A[n, 1].item()
        else:
            x = X_conv_B[n % 2].squeeze()
            kappa1 = kappas_conv_B[n % 2, 0].item()
            kappa2 = kappas_conv_B[n % 2, 1].item()

        img = imshow_grayscale(x, cmap=cmap)
        plt.gca().set_xticks([], [])
        plt.gca().set_yticks([], [])
        xlabel = (
            r"$\kappa_1 = {:.1f}$".format(kappa1)
            + ""
            + " $\kappa_2 = {:.1f}$".format(kappa2)
        )
        plt.xlabel(xlabel, fontsize=fontsize_xlabel)

    # https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
    if final_iteration is None:
        # generic designator
        final_iteration = "L"
    else:
        final_iteration = str(final_iteration)

    rows = [
        r"$p ( \bm{x} | \varphi^{(0)} )$",
        r"$p ( \bm{{x}} | \varphi^*_{{\mathcal{{M}},\mathcal{{D}}^{{({})}}}} )$".format(
            final_iteration
        ),
    ]

    for ax, row in zip(axi[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size="large", labelpad=10)  # size='large'

    if colorbar:
        grayscale_colorbar(img, fig, adjust=True, labels=labels)

    return fig, axi


def compare_microstructures_modified(
    factory,
    phi_init,
    phi_conv,
    N_examples=4,
    figsize=(9.5, 4.5),
    colorbar=False,
    final_iteration=None,
    multiphysics=False,
    cmap=None,
    labels=None,
):

    fig, axi = plt.subplots(2, N_examples, figsize=(9, 4.5))

    sampler = HmgWrappedSampler(factory)
    sampler.set_phi(phi_init)
    X_init, kappas_init = sampler.sample(N_examples)

    for n in range(N_examples):

        plt.sca(axi[0, n])
        x = X_init[n].squeeze()
        kappa1 = kappas_init[n, 0].item()
        kappa2 = kappas_init[n, 1].item()
        imshow_grayscale(x, cmap=cmap)
        plt.gca().set_xticks([], [])
        plt.gca().set_yticks([], [])
        plt.xlabel(r"$\kappa_1 = {:.2f}, \kappa_2 = {:.2f}$".format(kappa1, kappa2))

    sampler.set_phi(phi_conv)
    X_conv, kappas_conv = sampler.sample(N_examples)

    for n in range(N_examples):

        plt.sca(axi[1, n])
        x = X_conv[n].squeeze()
        kappa1 = kappas_conv[n, 0].item()
        kappa2 = kappas_conv[n, 1].item()
        img = imshow_grayscale(x, cmap=cmap)
        plt.gca().set_xticks([], [])
        plt.gca().set_yticks([], [])
        plt.xlabel(r"$\kappa_1 = {:.2f}, \kappa_2 = {:.2f}$".format(kappa1, kappa2))

    # https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
    if final_iteration is None:
        # generic designator
        final_iteration = "L"
    else:
        final_iteration = str(final_iteration)

    rows = [
        r"$p \left( \bm{x} \middle| \varphi^{(0)} \right)$",
        r"$p \left( \bm{{x}} \middle| \varphi^*_{{\mathcal{{M}},\mathcal{{D}}^{{({})}}}} \right)$".format(
            final_iteration
        ),
    ]

    for ax, row in zip(axi[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size="large", labelpad=10)  # size='large'

    if colorbar:
        grayscale_colorbar(img, fig, adjust=True, labels=labels)

    return fig, axi


def compare_microstructures(
    factory,
    phi_init,
    phi_conv,
    N_examples=4,
    figsize=(9.5, 4.5),
    colorbar=False,
    final_iteration=None,
):

    fig, axi = plt.subplots(2, N_examples, figsize=(9, 4.5))

    rf_plot, _ = factory.rf()
    rf_plot.set_phi(phi_init)

    for n in range(N_examples):

        plt.sca(axi[0, n])
        x = rf_plot.rsample().detach().cpu().numpy().squeeze()
        imshow_grayscale(x > factory.cutoff)
        plt.gca().set_xticks([], [])
        plt.gca().set_yticks([], [])

    rf_plot.set_phi(phi_conv)

    for n in range(N_examples):
        plt.sca(axi[1, n])

        x = rf_plot.rsample().detach().cpu().numpy().squeeze()
        img = imshow_grayscale(x > factory.cutoff)
        plt.gca().set_xticks([], [])
        plt.gca().set_yticks([], [])

    # https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots

    if final_iteration is None:
        # generic designator
        final_iteration = "L"
    else:
        final_iteration = str(final_iteration)

    rows = [
        r"$p \left( \bm{x} \middle| \varphi^{(0)} \right)$",
        r"$p \left( \bm{{x}} \middle| \varphi^*_{{\mathcal{{M}},\mathcal{{D}}^{{({})}}}} \right)$".format(
            final_iteration
        ),
    ]

    for ax, row in zip(axi[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size="large", labelpad=10)  # size='large'

    if colorbar:
        grayscale_colorbar(img, fig, adjust=True)

    return fig, axi


def binary_colormap():

    return colors.ListedColormap(["#1a1a1a", "#e6e6e6"])


def imshow_grayscale(x, cmap=None, **kwargs):

    x = x.squeeze()
    assert isinstance(x, np.ndarray) and x.ndim == 2

    cmap = binary_colormap() if cmap is None else cmap

    img = plt.imshow(x, cmap=cmap, **kwargs)

    return img


def grayscale_colorbar(
    img, fig=None, adjust=False, labels=None, cbar_title=r"\textbf{phases}"
):

    cmap = binary_colormap()
    bounds = [0 - 0.50, 0.5, 1 + 0.50]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    if adjust:
        assert fig is not None
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.017, 0.70])
        cbar = plt.colorbar(
            img,
            cax=cbar_ax,
            cmap=cmap,
            norm=norm,
            boundaries=bounds,
            ticks=[0 - 0.10, 1 + 0.10],
            spacing="proportional",
        )

    else:
        cbar = plt.colorbar(
            img,
            cmap=cmap,
            norm=norm,
            boundaries=bounds,
            ticks=[0, 1],
            spacing="proportional",
        )

    cbar.set_label(cbar_title, rotation=90, labelpad=-35)

    if labels is not None:
        cbar.ax.set_yticklabels(labels)

    return cbar


class JointDensityPlot(object):
    def __init__(self, phi):

        self._dim = None
        self.phi = phi

        self.cmap = plt.cm.magma

        self.wmodel = None
        self.hmg = None

    def __call__(self, *args, **kwargs):
        pass

    @classmethod
    def FromPhi(cls, phi: Union[list, np.ndarray], wmodel=None, hmg=None):

        if isinstance(phi, list):
            phi = np.stack(phi, 0)

        assert (
            isinstance(phi, np.ndarray)
            and phi.ndim == 2
            and phi.shape[1] == wmodel.rf.kernel.pdim
        )

        plotter = cls()
        plotter.hmg = hmg
        plotter.wmodel = wmodel

        return plotter

    def video(
        self,
        trajectory,
        N_samples=64,
        nbins: int = 50,
        fps: int = 5,
        path: str = None,
        mode: Literal["model", "reference"] = None,
        dpi=150,
        objective_representation=None,
        kwargs_or=None,
        xlabel=None,
        ylabel=None,
        title_generator=None,
        output_video=True,
        delete_images=False,
    ):

        # assert path is not None
        assert mode is not None and mode in ["model", "reference"]

        assert path is not None

        if mode == "model":
            trajectory.precompute_model(N_samples)
        elif mode == "reference":
            trajectory.precompute_ref(N_samples)
        else:
            raise RuntimeError

        kappa_min = trajectory.min(mode)
        kappa_max = trajectory.max(mode)

        video = MatplotlibVideo(path, framerate=fps)

        for n, PhiState in tqdm(enumerate(trajectory)):

            fig = plt.figure()

            title = title_generator(n) if title_generator is not None else None
            self.plot(
                PhiState,
                N_samples,
                nbins=nbins,
                kmin=kappa_min,
                kmax=kappa_max,
                mode=mode,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
            )

            if objective_representation is not None:
                kwargs_or = dict() if not kwargs_or else kwargs_or
                objective_representation.mark_2d(**kwargs_or)

            plt.savefig(
                path + "P_{}.png".format(n),
                dpi=dpi,
                bbox_inches="tight",
                transparent=True,
                pad_inches=0.08,
            )

            video.add_frame(fig, close_fig=True)

        video.save()

    def plot(
        self,
        phi_state,
        N_samples,
        nbins=100,
        title=None,
        xlabel=None,
        ylabel=None,
        colorbar=True,
        kmin=None,
        kmax=None,
        mode=None,
        **kwargs
    ):

        assert mode in ["reference", "model"]

        if mode == "reference":
            kappas = phi_state.demand_kappas_ref(N_samples)
        elif mode == "model":
            kappas = phi_state.demand_kappas_model(N_samples)
        else:
            raise RuntimeError

        X, Y, Z = self._density_2d(kappas, nbins=nbins, kmin=kmin, kmax=kmax)
        h1 = plt.pcolormesh(
            X, Y, Z.reshape(X.shape), shading="auto", cmap=self.cmap, **kwargs
        )
        if colorbar:
            plt.colorbar(h1)

        if title is not None:
            plt.title(title)

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

    # internal helper functions
    @classmethod
    def _establish_global_bounds(self, kappas) -> np.ndarray:

        assert (
            isinstance(kappas, list)
            or isinstance(kappas, np.ndarray)
            and kappas.ndim == 3
        )

        if isinstance(kappas, list):
            kappas = np.stack(kappas, 0)

        kappas = kappas.reshape(-1, kappas.shape[2])
        kappas_bounds = np.stack((kappas.min(0), kappas.max(0)), axis=1)

        return kappas_bounds

    @classmethod
    def _density_2d(cls, kappas: np.ndarray, nbins=100, offset=1, kmin=None, kmax=None):

        assert (
            isinstance(kappas, np.ndarray) and kappas.ndim == 2 and kappas.shape[1] == 2
        ), "density estimate failed"

        kmin = (
            np.stack(kappas).reshape(-1, kappas.shape[1]).min(0) - offset
            if kmin is None
            else kmin
        )
        kmax = (
            np.stack(kappas).reshape(-1, kappas.shape[1]).max(0) + offset
            if kmax is None
            else kmax
        )

        kde_ = kde.gaussian_kde([kappas[:, 0], kappas[:, 1]])
        xi, yi = np.mgrid[
            kmin[0] : kmax[0] : nbins * 1j, kmin[1] : kmax[1] : nbins * 1j
        ]
        zi = kde_(np.vstack([xi.flatten(), yi.flatten()]))

        return xi, yi, zi


def PlotDensities2DUtil(
    kappa_samples: list,
    nbins=300,
    mode="horizontal",
    kmin_offset=None,
    kmax_offset=None,
    bounds=None,
    colorbar=True,
    kmax_global=None,
    kmin_global=None,
    vmin=None,
    vmax=None,
    figsize=None,
    cmap=plt.cm.magma,
    filled=True,
):

    assert isinstance(kappa_samples, list)
    assert all(
        (isinstance(kappa, np.ndarray) and kappa.ndim == 2 and kappa.shape[1] == 2)
        for kappa in kappa_samples
    )
    assert isinstance(nbins, int) and nbins > 1
    assert mode in ["horizontal", "vertical"] or mode is None

    # number of 'snapshots'
    N_plots = len(kappa_samples)
    assert isinstance(N_plots, int) and N_plots >= 0

    # establish bounds
    kmin = np.stack(kappa_samples).reshape(-1, 2).min(0) - 1
    kmax = np.stack(kappa_samples).reshape(-1, 2).max(0) + 1

    # allow for offset
    if kmin_offset is not None:
        assert isinstance(kmin_offset, np.ndarray) and len(kmin_offset) == 2
        kmin = kmin - kmin_offset
    if kmax_offset is not None:
        assert isinstance(kmax_offset, np.ndarray) and len(kmax_offset) == 2
        kmin = kmin + kmax_offset

    if kmin_global is not None:
        kmin = np.ones(2) * kmin_global

    if kmax_global is not None:
        kmax = np.ones(2) * kmax_global

    if mode is None:
        assert N_plots == 1
        fig, axi = None, None
    elif mode.lower() == "horizontal":

        figsize = figsize = (4.65 * N_plots, 4) if figsize is None else figsize
        fig, axi = plt.subplots(1, N_plots, figsize=figsize)

    elif mode.lower() == "vertical":

        figsize = (5.8, 5.0 * N_plots) if figsize is None else figsize
        fig, axi = plt.subplots(N_plots, 1, figsize=figsize)

    h = list()
    for n in range(N_plots):

        if N_plots > 1:
            ax = axi[n]
        else:
            ax = axi

        x = kappa_samples[n][:, 0]
        y = kappa_samples[n][:, 1]

        kde_ = kde.gaussian_kde([x, y])
        xi, yi = np.mgrid[
            kmin[0] : kmax[0] : nbins * 1j, kmin[1] : kmax[1] : nbins * 1j
        ]
        zi = kde_(np.vstack([xi.flatten(), yi.flatten()]))

        if mode is not None:
            plt.sca(ax)
        if filled:
            h1 = plt.contourf(
                xi, yi, zi.reshape(xi.shape), cmap=cmap, vmin=vmin, vmax=vmax
            )
        else:
            h1 = plt.contour(
                xi, yi, zi.reshape(xi.shape), cmap=cmap, vmin=vmin, vmax=vmax
            )
        h.append(h1)

        if bounds is not None:

            assert isinstance(bounds, np.ndarray) and bounds.shape == (2, 2)
            midpoint = tuple(bounds.mean(1))
            extensions = bounds[:, 1] - bounds[:, 0]
            xy = np.array(midpoint) - 0.5 * extensions
            xy = tuple(xy)  # lower left corner, anchor point
            rect = patches.Rectangle(
                xy,
                extensions[0],
                extensions[1],
                linewidth=1.5,
                edgecolor="g",
                facecolor="none",
            )
            plt.gca().add_patch(rect)

        if colorbar:
            if mode is not None and mode.lower() == "vertical":

                fig.colorbar(h1)

    if colorbar:

        if mode is None:
            pass

        elif mode.lower() == "horizontal":
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(h1, cax=cbar_ax)

    else:
        raise NotImplementedError("Currently can only do horizontal mode plot")

    return fig, axi, h
