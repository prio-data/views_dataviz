"""Radar plots."""

from typing import Optional, Union, Tuple, List
from math import pi
import matplotlib.pyplot as plt
import pandas as pd


def plot_radar(
    df: pd.DataFrame,
    categories: str,
    entries: List[str],
    minmax: Optional[Tuple[float, float]] = None,
    legend_labels: List[str] = None,
    legend_y: float = 1,
    figsize: Tuple[int, int] = (6, 6),
    tick_locations: Optional[List[float]] = None,
    tick_labels: Optional[List[str]] = None,
    tick_size: int = 12,
    title: Optional[str] = None,
    titlesize: int = 14,
    cmap: str = "tab10",
    colors: List[str] = None,
    fill: bool = False,
    linewidth: Union[float, List[float]] = 1,
    linestyle: Union[str, List[str]] = "solid",
    marker: Union[str, List[str]] = None,
    alpha: Union[float, List[float]] = 1,
    path: Optional[str] = None,
):
    """Plots a radar chart.

    Args:
        df: Pandas DataFrame containing the data to plot.
        categories: Column name containing the categories.
        entries: List of column names to plot on the radar.
        minmax: Tuple that sets min and max of values plotted.
        legend_labels: Optional list of legend labels.
        legend_y: Legend location in axes coords.
        figsize: Tuple of width and height in inches.
        tick_locations: Optional locations of ticks plotted on radar.
        tick_labels: Optional labels of ticks plotted on radar.
        tick_size: Textsize of ticks.
        title: Optional title to add to figure.
        titlesize: Textsize of title.
        cmap: String referring to a matplotlib colormap. Colors are assigned 
            to each enumerated entry: cmap(i).
        colors: Optional list of colors. Overrides cmap. Example:
            ["#808080", "blue"]
        fill: Bool indicating whether to fill the polygons plotted.
        linewidth: A single float for the width of lines plotted, or a list
            containing the widths to apply per entry.
        linestyle: A single string linestyle to apply to the lines plotted, or
            a list containing the linestyles to apply per entry.
        marker: Optional markerstyle or list of markerstyles for each entry.
        alpha: Alpha of fill.
        path: Optional. Writes to path if set.
    """
    _, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    nodes = len(df[categories])

    # Determine the angle of each axis in the plot.
    angles = [n / float(nodes) * 2 * pi for n in range(nodes)]
    angles += angles[:1]

    # Draw one tick per var.
    plt.xticks(angles[:-1], df[categories], size=tick_size)

    # Draw ylabels.
    ax.set_rlabel_position(0)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Adjust yticks and ylims.
    plt.yticks(
        ticks=tick_locations, labels=tick_labels, color="grey", size=tick_size
    )
    if minmax is not None:
        plt.ylim(minmax[0], minmax[1])

    # Set up radardata, repeating the first value to close the circular graph.
    radardata = {}
    for col in entries:
        coldata = df[col].to_list()
        coldata += coldata[:1]
        radardata[col] = coldata

    # Set up color in style specs.
    style = {}
    for i, col in enumerate(entries):
        if colors is not None:
            style[col] = {"color": colors[i]}
        else:
            cmap = plt.get_cmap(cmap)
            style[col] = {"color": cmap(i)}

    # Set up other style specs.
    for i, col in enumerate(entries):
        style[col]["linewidth"] = (
            linewidth[i] if isinstance(linewidth, list) else linewidth
        )
        style[col]["linestyle"] = (
            linestyle[i] if isinstance(linestyle, list) else linestyle
        )
        style[col]["alpha"] = alpha[i] if isinstance(alpha, list) else alpha
        style[col]["marker"] = (
            marker[i] if isinstance(marker, list) else marker
        )

    # Plot.
    for col in radardata:
        ax.plot(
            angles,
            radardata[col],
            linewidth=style[col]["linewidth"],
            linestyle=style[col]["linestyle"],
            color=style[col]["color"],
            alpha=style[col]["alpha"],
            marker=style[col]["marker"],
            label=col,
        )
        if fill:
            ax.fill(
                angles, radardata[col], color=style[col]["color"], alpha=0.1
            )

    if legend_labels is not None:  # Override col-as-label default.
        ax.legend(
            legend_labels, bbox_to_anchor=(1.08, legend_y), frameon=False
        )
    else:
        ax.legend(bbox_to_anchor=(1.08, legend_y), frameon=False)

    if title:
        ax.set_title(title, fontdict={"fontsize": titlesize})

    ax.spines["polar"].set_visible(False)

    if path is not None:
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
