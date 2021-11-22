"""Dumbbell plot"""

from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt
import pandas as pd


def plot_dumbbell(
    df: pd.DataFrame,
    current: str,
    previous: str,
    labels: str,
    axislabel: str,
    title: str,
    orientation: str = "vertical",
    markersize: int = 100,
    markercolors: Tuple[int, int] = ("black", "red"),
    linecolor: str = "grey",
    lims: Optional[Tuple[int, int]] = None,
    pad: float = 0.2,
    figsize: Tuple[float, float] = (10.0, 12.0),
    legend_kwds: Optional[Dict] = None,
    ascending: bool = False,
    path: Optional[str] = None,
):
    """Plots vertical or horizontal dumbbell chart for selected deltas.

    Args:
        df: pd.DataFrame containing the current and previous series.
        current: Column name of current series.
        previous: Column name of previous series.
        labels: Column name of associated labels.
        axislabel: Label to give to the axis (x for vertical, y for horizontal).
        title: Title to the figure.
        orientation: String for orientation: "vertical" or "horizontal".
        markersize: Size of markers in scatter.
        markercolors: Tuple of colors for the previous and current markers.
        linecolor: Color for the lines connecting the scatter markers.
        lims: Optional tuple of vmin, vmax of x-axis if vertical, y-axis if
            horizontal.
        pad: Float for padding between outer frame and the first and last
            dumbbell.
        figsize: Tuple of (width, height) to pass as figsize.
        legend_kwds: Optional dictionary containing keyword arguments for
            additional customization of the legend.
        ascending: Bool determinining whether to sort the delta ascending or
            descending.
        path: Path to write figure to.
    """
    # Sort by delta and reindex.
    current, previous = df[current], df[previous]
    labels = df[labels]
    delta = current - previous
    delta = delta.sort_values(ascending=ascending)

    current = current.reindex(delta.index)
    previous = previous.reindex(delta.index)
    labels = labels.reindex(delta.index)
    y_range = range(1, len(delta.index) + 1)

    _, ax = plt.subplots(figsize=figsize)
    if orientation not in ("vertical", "horizontal"):
        raise RuntimeError(f"{orientation} is not a valid orientation.")
    if orientation == "vertical":
        plt.hlines(
            y=y_range,
            xmin=current,
            xmax=previous,
            color=linecolor,
            lw=1.5,
            zorder=1,
        )
        plt.scatter(
            previous,
            y_range,
            color=markercolors[0],
            label=previous.name,
            s=markersize,
            zorder=2,
        )
        plt.scatter(
            current,
            y_range,
            color=markercolors[1],
            label=current.name,
            s=markersize,
            zorder=2,
        )
    else:
        plt.vlines(
            x=y_range,
            ymin=current,
            ymax=previous,
            color=linecolor,
            lw=1.5,
            zorder=1,
        )
        plt.scatter(
            y_range,
            previous,
            color=markercolors[0],
            label=previous.name,
            s=markersize,
            zorder=2,
        )
        plt.scatter(
            y_range,
            current,
            color=markercolors[1],
            label=current.name,
            s=markersize,
            zorder=2,
        )

    axis = "x" if orientation == "vertical" else "y"
    ax.grid(
        which="major",
        axis=axis,
        linestyle="--",
        dashes=(2, 3),
        lw=1,
        color="grey",
        alpha=0.5,
    )
    ax.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.margins(pad)

    if lims is not None:
        vmin, vmax = lims
        if orientation == "vertical":
            plt.xlim(vmin, vmax)
        else:
            plt.ylim(vmin, vmax)

    if orientation == "vertical":
        plt.yticks(y_range, labels, size=11)
        plt.xlabel(axislabel, size=14, labelpad=20)
    else:
        plt.xticks(y_range, labels, size=11)
        plt.ylabel(axislabel, size=14, labelpad=20)

    if legend_kwds is not None:
        plt.legend(**legend_kwds)
    else:
        plt.legend()

    plt.title(title, loc="left", size=15, pad=25)
    if path is not None:
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
