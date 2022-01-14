"""Error plots for continuous outcome variables"""

from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as pltcol
from matplotlib.patches import ConnectionPatch

from views_dataviz.map.presets import ViewsMap


# TODO: limit assumes positive values. Fix for a range that is only negative.


class ContinuousCriticismPlot:
    """
    Generates a model criticism plot for continuous outcome variables.

    Attributes
    ----------
    df: pd.DataFrame containing the predictions, actuals and labels.
    y_true: Column name for the actual observations.
    y_pred: Column name for the model predictions.
    lab: Column name for the row labels.
    figsize: Tuple of (width, height) to pass to figsize.
    framesize: Linewidth of figure frame.
    markersize: Size of markers in scatter plot.
    n_worst: Number of worst error annotations to plot.
    cmap: String referring to one maplotlib colormaps.
    delta_alpha: Bool indicating whether to add an alpha channel based
        on the deltas.
    title: String title to give to figure. Title is empty by default.
    notation: Label for the outcome variable.
    titlesize: Textsize of figure title.
    labelsize: Textsize of xlabel and ylabel.
    annotsize: Textsize of annotations.
    limit: Upper end of the xlimit. The plot centers on zero.
    y_margin: Margin to apply to the y-axis. The margin added to each limit of
        the Axes is the margin times the data interval.
    x_margin: Margin to apply to the x-axis. The margin added to each limit of
        the Axes is the margin times the data interval.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y_pred: str,
        y_true: str,
        lab: str,
        figsize: Tuple[float, float] = (5.0, 7.0),
        framesize: float = 2,
        markersize: float = 100.0,
        n_worst: int = 5,
        cmap: str = "coolwarm",
        delta_alpha: bool = False,
        title: str = "",
        notation: str = "y",
        titlesize: float = 16.0,
        labelsize: float = 12.0,
        ticksize: float = 10.0,
        annotsize: float = 10.0,
        limit: Tuple[float, float] = None,
        x_margin: float = 0.1,
        y_margin: float = 0.05,
        path: str = None,
    ):
        self.df = df
        self.y_pred = y_pred
        self.y_true = y_true
        self.lab = lab
        self.figsize = figsize
        self.framesize = framesize
        self.markersize = markersize
        self.n_worst = n_worst
        self.cmap = cmap
        self.delta_alpha = delta_alpha
        self.title = title
        self.notation = notation
        self.titlesize = titlesize
        self.labelsize = labelsize
        self.ticksize = ticksize
        self.annotsize = annotsize
        self.limit = limit
        self.x_margin = x_margin
        self.y_margin = y_margin
        self.path = path

        # Build figure on init.
        (
            self.prepare_data(
                df=self.df,
                y_true=self.y_true,
                y_pred=self.y_pred,
                lab=self.lab,
            )
            .prepare_figure()
            .scatter()
            .rug()
            .connect_rug()
            .reg()
            .annotate()
        )

    def __str__(self):
        return f"Continuous criticism plot of {self.title}."

    def __repr__(self):
        return f"Continuous criticism plot of {self.title}."

    def prepare_data(self, df, y_true, y_pred, lab):
        """Prepare data for plot."""
        delta = df[y_pred] - df[y_true]
        self.s_pred = df[y_pred].sort_values()
        self.s_lab = df[lab].reindex(self.s_pred.index).reset_index(drop=True)
        self.s_obs = (
            df[y_true].reindex(self.s_pred.index).reset_index(drop=True)
        )
        self.delta = delta.reindex(self.s_pred.index).reset_index(drop=True)
        self.s_pred = self.s_pred.reset_index(drop=True)
        self.worst_op = delta[-self.n_worst :].sort_index(ascending=False)
        self.worst_up = delta[0 : self.n_worst].sort_index(ascending=False)
        return self

    def prepare_figure(self):
        """Set figure frame."""
        self.fig, self.axs = plt.subplots(
            nrows=2,
            figsize=self.figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1]},
        )
        plt.subplots_adjust(hspace=0)
        # Thicken frame.
        for axis in ["top", "bottom", "left", "right"]:
            self.axs[0].spines[axis].set_linewidth(self.framesize)
            self.axs[1].spines[axis].set_linewidth(self.framesize)
        # Set xlims (adding space for the rugplot).

        # Datalims + margin
        if self.limit is not None:
            self.limit = (1 + self.x_margin) * self.limit
            plt.xlim(-self.limit, self.limit)
        else:
            vmin, vmax = self.s_pred.min(), self.s_pred.max()
            self.limit = (1 + self.x_margin) * max(abs(vmin), abs(vmax))
            plt.xlim(-self.limit, self.limit)

        # Set up colorscheme.
        self.cmap = plt.get_cmap(self.cmap)
        self.norm = plt.Normalize(self.s_obs.min(), self.s_obs.max())
        self.colors = self.cmap(self.norm(self.s_obs))
        # Set up alphas if requested.
        if self.delta_alpha:
            dalpha = abs(self.delta.copy())
            # Normalize that 0-1 so it can be used as alpha.
            alphas = (dalpha - np.min(dalpha)) / (
                np.max(dalpha) - np.min(dalpha)
            )
            alphas.loc[alphas < 0.3] = 0.3  # Set alpha 0.3 as the minimum.
            self.colors[
                :, 3
            ] = alphas  # Alpha is the fourth col in a color array.
        self.axs[0].set_title(self.title, size=self.titlesize, pad=20)
        return self

    def scatter(self):
        """Plot figure."""
        # Set up top ax.
        self.axs[0].scatter(
            self.s_pred,
            self.s_pred.index,
            color=self.colors,
            s=self.markersize,
        )
        self.axs[0].margins(self.y_margin)
        self.axs[0].grid(
            which="major",
            axis="x",
            lw=1,
            color="black",
            alpha=0.1,
        )
        self.axs[0].set_ylabel(
            "Observation (ordered by {})".format("{}".format(self.notation)),
            size=self.labelsize,
        )
        self.axs[0].tick_params(labelsize=self.ticksize)
        # Draw hline at first index where our predictions pass zero.
        positive_mask = self.s_pred >= 0
        zero_index = positive_mask.idxmax()
        self.axs[0].axhline(
            y=zero_index, xmin=0, xmax=1, color="grey", lw=0.8
        )  # axhline uses axes coordinate system, so just 0-1.
        self.axs[0].axvline(
            x=0, ymin=0, ymax=len(self.s_pred), color="grey", lw=0.8
        )
        return self

    def rug(self):
        """Add rug to top axis."""
        rax = self.axs[0].inset_axes(bounds=[0.95, 0, 0.05, 1], zorder=1)
        for index, value in self.s_obs.items():
            edgecolor = pltcol.to_hex(self.cmap(self.norm(value)))
            rax.hlines(y=index, xmin=0, xmax=1, color=edgecolor, alpha=1)
        rax.set_xticks([])
        rax.set_yticks([])
        rax.margins(self.y_margin)  # Equivalent to axs[0].
        rax.axis("off")
        return self

    def reg(self):
        """Add regplot to bottom axis."""
        sns.regplot(
            x=self.s_pred,
            y=self.s_obs,
            ax=self.axs[1],
            lowess=False,
            line_kws={
                "color": "black",
                "linewidth": 1,
                "linestyle": "dashed",
                "alpha": 0.5,
            },
            scatter_kws={"color": self.colors},
        )
        self.axs[1].margins(0.2)
        self.axs[1].grid(
            which="major",
            axis="x",
            lw=1,
            color="black",
            alpha=0.2,
        )
        self.axs[1].set_xlabel(
            "Prediction ({})".format("{}".format(self.notation)),
            size=self.labelsize,
        )
        self.axs[1].set_ylabel(
            "Observed {}".format("{}".format(self.notation)),
            size=self.labelsize,
        )
        self.axs[1].tick_params(labelsize=self.ticksize)
        return self

    def connect_rug(self):
        """Connect scatter to rug."""
        for index in self.worst_op.keys():
            edgecolor = pltcol.to_hex(self.cmap(self.norm(self.s_obs[index])))
            self.axs[0].hlines(
                y=index,
                xmin=self.s_pred[index],
                xmax=self.limit,
                color=edgecolor,
                alpha=0.2,
            )
        for index in self.worst_up.keys():
            edgecolor = pltcol.to_hex(self.cmap(self.norm(self.s_obs[index])))
            self.axs[0].hlines(
                y=index,
                xmin=self.s_pred[index],
                xmax=self.limit,
                color=edgecolor,
                alpha=0.2,
            )
        return self

    def annotate(self):
        """Add label annotations."""
        # Axis coords x, data coords y.
        trans = self.axs[0].get_yaxis_transform()
        delta = self.delta.sort_values()
        spacing = len(delta) / 20
        step = 0.0
        start_loc = len(delta)

        # Hang annotations for positive deltas.
        for index in self.worst_op.keys():
            # Add label above the annotations.
            if step == 0:
                self.axs[0].annotate(
                    "Overpredicted",
                    xy=(self.limit, index),
                    xycoords="data",
                    xytext=(1.15, start_loc),  # A bit above list.
                    textcoords=trans,
                    size=self.annotsize - 2,
                )
                step = step + spacing
            # Set up color, horizontal lines, annotations.
            edgecolor = pltcol.to_hex(self.cmap(self.norm(self.s_obs[index])))
            self.axs[0].annotate(
                self.s_lab[index],  # Get associated label by index.
                xy=(self.limit, index),
                xycoords="data",
                xytext=(1.15, start_loc - step),  # start_loc - step
                textcoords=trans,
                va="center",
                ha="left",
                size=self.annotsize,
            )
            # Little trick here to actually attach to the left center point.
            self.axs[0].annotate(
                "",
                xy=(self.limit, index),
                xycoords="data",
                xytext=(1.15, start_loc - step),  # start_loc - step
                textcoords=trans,
                arrowprops=dict(
                    arrowstyle="-", edgecolor=edgecolor, shrinkB=0, shrinkA=0
                ),
            )
            step = step + spacing

        # Stack annotations for negatives deltas.
        step = 0.0
        start_loc = start_loc - ((self.n_worst + 2) * spacing)
        for index in self.worst_up.keys():
            # Add label above the annotations.
            if step == 0:
                self.axs[0].annotate(
                    "Underpredicted",
                    xy=(self.limit, index),
                    xycoords="data",
                    xytext=(
                        1.15,
                        start_loc,
                    ),  # Place a bit above list.
                    textcoords=trans,
                    size=self.annotsize - 2,
                )
                step = step + spacing
            # Draw line to right axis and to annotation base.
            edgecolor = pltcol.to_hex(self.cmap(self.norm(self.s_obs[index])))
            # Set up color, horizontal lines, annotations.
            self.axs[0].annotate(
                self.s_lab[index],  # Get associated label by index.
                xy=(self.limit, index),
                xycoords="data",
                xytext=(1.15, start_loc - step),
                textcoords=trans,
                va="center",
                ha="left",
                size=self.annotsize,
            )
            # Little trick here to actually attach to the left center point.
            self.axs[0].annotate(
                "",
                xy=(self.limit, index),
                xycoords="data",
                xytext=(1.15, start_loc - step),
                textcoords=trans,
                arrowprops=dict(
                    arrowstyle="-", edgecolor=edgecolor, shrinkB=0, shrinkA=0
                ),
            )
            step = step + spacing
        return self

    def save(self, path, dpi=200, **kwargs):
        """Save the figure."""
        self.fig.savefig(path, dpi=dpi **kwargs)
        plt.close(self.fig)


class ContinuousCriticismMap(ContinuousCriticismPlot):
    """
    Extends ContinuousCriticismPlot with a mapping method. Overrides original
    labelling method. Observations are matched with *centroid* locations in the
    gdf by the polygon identifier provided in `lab`. Make sure that this column
    is the same in df as in your gdf (e.g. "pg_id").

    Attributes
    ----------
    df: pd.DataFrame containing the predictions, actuals and labels.
    y_true: Column name for the actual observations.
    y_pred: Column name for the model predictions.
    lab: Column name for the row labels.
    mc_kwargs: Dictionary of keyword arguments used by ModelCriticismPlot.
    map_kwargs: Dictionary of keyword arguments used by ViewsMap.
    title: Optional title.
    titlesize: Textsize of title.
    """

    def __init__(
        self,
        df,
        y_true,
        y_pred,
        lab,
        mc_kwargs,
        map_kwargs,
        title="",
        titlesize=18,
    ):
        super().__init__(df, y_pred, y_true, lab, **mc_kwargs)
        self.set_map_ax()
        self.mapper = ViewsMap(figure=(self.fig, self.map_ax), **map_kwargs)
        self.fig.suptitle(title, x=0, size=titlesize)

    def annotate(self):
        """Suppresses the annotation method in the parent class."""
        return self

    def set_map_ax(self):
        """Add an ax for the map."""
        self.map_ax = self.fig.add_axes([1.1, 0, 1.1, 1])
        self.map_ax.tick_params(
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
        )
        for axis in ["top", "bottom", "left", "right"]:
            self.map_ax.spines[axis].set_linewidth(self.framesize)

    def plot_map_layer(self, suppress_colorbar=True, **kwargs):
        """
        Plots a map layer.

        Parameters
        ----------
        suppress_colorbar: Bool determining whether to plot the colorbar.
            Defaults to True.
        **kwargs: Dictionary of Mapper.add_layer keyword arguments.
        """
        self.mapper.add_layer(**kwargs)
        if suppress_colorbar:
            if hasattr(self.mapper, "cax"):
                self.mapper.cax.remove()  # Remove default colorbar.
        return self

    def connect(self, gdf):
        """
        Finds the appropriate map coordinate by matching labs.

        Parameters
        ----------
        gdf: gpd.GeoDataFrame to plot.
        """
        for idx in self.worst_op.keys():
            # For some reason geopandas loses some information when queried.
            mapindex = gdf.loc[gdf[self.lab] == self.s_lab[idx]].index[0]
            mapcoord = gdf.loc[mapindex].geom.centroid.coords[0]
            edgecolor = pltcol.to_hex(self.cmap(self.norm(self.s_obs[idx])))
            con = ConnectionPatch(
                xyA=(self.limit, idx),
                xyB=mapcoord,
                coordsA="data",
                coordsB="data",
                axesA=self.axs[0],
                axesB=self.map_ax,
                color=edgecolor,
            )
            self.map_ax.add_artist(con)
        for idx in self.worst_up.keys():
            mapindex = gdf.loc[gdf[self.lab] == self.s_lab[idx]].index[0]
            mapcoord = gdf.loc[mapindex].geom.centroid.coords[0]
            edgecolor = pltcol.to_hex(self.cmap(self.norm(self.s_obs[idx])))
            con = ConnectionPatch(
                xyA=(self.limit, idx),
                xyB=mapcoord,
                coordsA="data",
                coordsB="data",
                axesA=self.axs[0],
                axesB=self.map_ax,
                color=edgecolor,
            )
            self.map_ax.add_artist(con)
        return self
