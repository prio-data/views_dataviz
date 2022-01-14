"""Ablation plot"""

from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_ablation(
    df: pd.DataFrame,
    primary_cols: List[str],
    secondary_cols: List[str],
    lab: str,
    sort_by: Optional[str] = None,
    cmap: str = "viridis",
    suptitle: Optional[str] = None,
    titles: Optional[List[str]] = None,
    x_labels: Optional[List[str]] = None,
    x_labels_rotation: int = 30,
    x_offset: Optional[float] = None,
    barwidth: float = 0.5,
    titlesize: int = 18,
    ticksize: int = 12,
    labelsize: int = 15,
    ylabel: Optional[str] = None,
    ylabelpad: float = 0.05,
    cbarlabel: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (5, 5),
    path: Optional[str] = None,
):
    """Plots an ablation chart.

    Used to plot model ablation results, but could also be useful elsewhere.
        Plots model performance by category (e.g. step) and two dimensions (one on
        barchart, one on color).

    Args:
        df: pd.DataFrame containing data to plot.
        primary_cols: List of column names for the first metric to plot.
            This will determine bar length.
        secondary_cols: List of column names for the second metric to plot.
            This will determine the color.
        lab: Column name for the label, such as the modelname.
        sort_by: Column name to sort by. Defaults to the first primary column.
        cmap: String for Matplotlib colormap to apply to the figure.
        suptitle: Optional suptitle to add to the figure.
        titles: List of titles to give to each respective subplot.
        x_labels: List of labels to give to each respective subplot's x-axis.
        x_offset: Offset from zero to apply to the xlimits. If not provided,
            it will find the maximum absolute value, and multiply by 1.05.
        barwidth: Width of bars.
        titlesize: Textsize for the titles.
        ticksize: Textsize for the ticks.
        labelsize: Textsize for the labels.
        ylabel: Optional y-label (left-hand side) to add.
        ylabelpad: Padding from the figure frame to add to the ylabel.
        cbarlabel: Optional label to add to the colorbar.
        vmin: Optional vmin to apply to the color range.
        vmax: Optional vmax to apply to the color range.
        figsize: Tuple containing width and height in inches.
        path: Optional, writes to path if set.
    """
    if len(primary_cols) != len(secondary_cols):
        raise RuntimeError(
            "Number of primary columns does not match number of secondary columns."
        )
    n_steps = len(primary_cols)
    fig, (axes) = plt.subplots(1, n_steps, sharey=True, figsize=figsize)
    if n_steps == 1:
        axes = [axes]

    # Set tight layout.
    plt.subplots_adjust(wspace=0.05, hspace=0, left=0)

    # Get colormap object from cmap.
    cmap = plt.get_cmap(cmap)

    # Sort values by first column.
    if sort_by is None:
        df = df.sort_values(by=primary_cols[0])
    else:
        df = df.sort_values(by=sort_by)

    # Set up min/max and scaler.
    if vmin is None:
        vmin = df[secondary_cols].min().min()
    if vmax is None:
        vmax = df[secondary_cols].max().max()
    rescale = lambda y: (y - vmin) / (vmax - vmin)

    # Set up axes.
    for i, ax in enumerate(axes):
        ax.axvline(x=0, color="lightgrey", zorder=0)
        ax.barh(
            df[lab],
            df[primary_cols[i]],
            color=cmap(rescale(df[secondary_cols[i]])),
            zorder=3,
            height=0.5,
        )
        # Plot left - right lines.
        ax.grid(
            color="lightgrey", linestyle="-", linewidth=1, axis="y", zorder=4
        )
        if i > 0:
            ax.yaxis.set_tick_params(length=0)

        # Limit the x_axis as determined by offset.
        if x_offset is None:
            x_offset = 1.05 * max(
                abs(df[primary_cols].min().min()),
                df[primary_cols].max().max(),
            )
        x_min, x_max = 0 - x_offset, 0 + x_offset
        ax.set_xlim([x_min, x_max])

        # Set ax title.
        if titles is not None:
            if len(titles) != len(primary_cols):
                raise RuntimeError(
                    "Number of titles does not match the number of subplots."
                )
            ax.set_title(titles[i], fontsize=titlesize, pad=10)
        else:
            ax.set_title(primary_cols[i], fontsize=titlesize, pad=10)

        # Set ax label.
        if x_labels is not None:
            ax.set_xlabel(x_labels[i], fontsize=labelsize)
        else:
            ax.set_xlabel("Ablation loss", fontsize=labelsize)

        # Set x-ticks.
        ax.set_xticks(ax.get_xticks()[1:-1])
        ax.set_xticklabels(
            [np.round(i, 3) for i in ax.get_xticks()],
            rotation=x_labels_rotation,
        )

        # Set x and y axis tick label size.
        ax.tick_params(axis="both", which="major", labelsize=ticksize)

    # Get bounding box information for the axes of the last subplot.
    bbox_ax0 = axes[0].get_position()

    # Write models/features as the Y axis label.
    if ylabel is not None:
        fig.text(
            bbox_ax0.x0 - ylabelpad,
            0.5,
            ylabel,
            va="center",
            rotation="vertical",
            fontsize=labelsize,
        )

    # Make ax for colorbar and add to canvas.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.1)

    # Fill in the colorbar and adjust the ticks.
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm._A = []  # pylint: disable=protected-access
    cbar = plt.colorbar(sm, cax=cax)
    if cbarlabel is not None:
        cbar = cbar.set_label(
            cbarlabel,
            horizontalalignment="center",
            fontsize=labelsize,
            labelpad=10,
        )
    cax.tick_params(labelsize=ticksize)

    if suptitle is not None:
        fig.suptitle(
            suptitle,
            fontsize=titlesize,
            y=1.05,
            x=0.5,
            horizontalalignment="right",
            verticalalignment="top",
        )

    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
