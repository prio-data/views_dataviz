"""Heatmap plotting functions."""

from typing import List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
import matplotlib.colors as mpcol


def plot_heatmap(
    df: pd.DataFrame,
    s: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    cmap: Any = "viridis",
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    tick_values: Optional[List[float]] = None,
    tick_labels: Optional[List[str]] = None,
    rotation_xlabels=None,
    textsize: float = 15,
    figsize: Tuple[float, float] = (10.0, 10.0),
    legend: bool = True,
    boundaries: Optional[List[float]] = None,
    categorical: bool = False,
    path: Optional[str] = None,
) -> None:
    """Plot a heatmap from a (MultiIndexed) Pandas dataframe.

    Args:
        df: pd.DataFrame containing the series to plot.
        s: Column name for the series to plot.
        x: Optional column name for the x-labels (usually time). Note that if
            it is non-integer, this needs to be datetime for it to sort
            correctly.
        y: Optional column name for the group labels.
        colors: Optional list of colors to apply if the s are categorical.
        title: Title to add to figure.
        cmap: Matplotlib colormap to use.
        ymin: Minimum to map the data to.
        ymax: Maximum to map the data to.
        tick_values: List of selected ticks to show in colorbar.
        tick_values: List of strings for the selected ticks.
        rotation_xlabels: Degree of rotation applied to the xlabels.
        textsize: Base size of text. Title/ticks are hardcoded relatives.
        figsize: Tuple (width, height) to pass as figure size.
        legend: Bool determining whether to draw a legend.
        boundaries: Optional boundaries to apply to colorscale.
        categorical: Bool to indicate the data is categorical.
        path: Optional. Write to path if set.
    """
    # Take the index values as default axes of the matrix.
    try:
        timevar = x if x is not None else df.index.names[0]
        groupvar = y if y is not None else df.index.names[1]
    except IndexError as e:
        msg = "Series is not xy MultiIndexed. Try providing x and y."
        raise Exception(msg) from e

    # Check for series dtype. Factorize if not numeric.
    if not pd.api.types.is_numeric_dtype(df[s]):
        warnings.warn("Series does not look numeric. Factorizing.")
        df[s] = pd.factorize(df[s], sort=True)[0]

    # Set vmin, vmax.
    vmin = df[s].min() if not ymin else ymin
    vmax = df[s].max() if not ymax else ymax

    # Pivot df.
    df_matrix = df.reset_index().pivot(
        index=groupvar, columns=timevar, values=s
    )

    # Set size.
    plt.figure(figsize=figsize)

    # Plot.
    if colors is not None:
        cmap = mpcol.LinearSegmentedColormap.from_list("categorical", colors)
    ax = sns.heatmap(
        df_matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=1,
        cbar=False,
    )
    plt.title(title, loc="left", size=textsize + 5, pad=20)

    # Remove axes labels.
    ax.set_ylabel("")
    ax.set_xlabel("")

    plt.yticks(fontsize=textsize - 5, va="center")
    plt.xticks(fontsize=textsize - 5, rotation=rotation_xlabels, ha="center")

    # Make ax for colorbar and add to canvas.
    if legend:
        if categorical:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cax.axis("off")
            # Reconstruct color map categorically.
            # Ticks are the unique values in s or specifically provided ones.
            ticks = df[s].unique() if tick_values is None else tick_values
            if colors is None:
                colorf = plt.get_cmap(cmap)
            else:
                colorf = mpcol.LinearSegmentedColormap.from_list(
                    "categorical", colors
                )
            colors = colorf(np.linspace(0, 1, len(ticks)))
            # Alter labels in same order if requested.
            labels = (
                sorted(df[s].unique()) if tick_labels is None else tick_labels
            )
            # Make patches according to categorical colors.
            patches = [
                mpatches.Patch(facecolor=c, edgecolor=c) for c in colors
            ]
            legend = cax.legend(
                patches,
                labels,  # Sorted unique maps to pd.factorize.
                handlelength=0.75,
                loc="upper left",
                frameon=False,
            )
            for text in legend.get_texts():
                text.set_ha("left")
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            # Fill in the colorbar and adjust the ticks.
            if boundaries is not None:
                norm = mpcol.BoundaryNorm(boundaries=boundaries, ncolors=256)
            else:
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []  # pylint: disable=protected-access
            cbar = plt.colorbar(
                sm,
                cax=cax,
                ticks=tick_values,
            )
            cax.tick_params(labelsize=textsize)

            # Assume ticks are labels if only values are provided.
            if tick_labels is not None:
                if tick_values is not None:
                    cbar.set_ticklabels(tick_labels)
                else:
                    raise RuntimeError("Need tick values to match labels to.")

    # Finish up and save.
    if path is not None:
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
