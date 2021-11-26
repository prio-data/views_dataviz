"""Parallel coordinates plot"""

from typing import List, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_parallel_coordinates(
    df: pd.DataFrame,
    columns: List[str],
    linecolors: Dict[str, str] = None,
    linestyles: Dict[str, str] = None,
    reverse: List[str] = None,
    figsize: Tuple[float, float] = (10.0, 5.0),
    labelsize: float = 12.0,
    titlesize: float = 16.0,
    legend_anchor: Tuple[float, float] = (1.2, 1.0),
    lw: float = 2.0,
    path: str = None,
):
    """Generates a parallel coordinates plot.

    Parameters
    ----------
    df: pd.DataFrame containing the scores to plot, indexed on the subject
        (models for example).
    columns: List or column names for the metrics to plot.
    linecolors: Dictionary of index: color pairs to overwrite. Colors
        default to tab10. For example: {"model_a": "green", "model_b": "red"}.
    linestyles: Dictionary of column: style pairs to overwrite. Styles
        default to "solid". For example: {"model_a": "solid", "model_b": "dotted"}.
    reverse: Optional list of column names to reverse to descending descending scale.
    figsize: Tuple of figure size in inches.
    labelsize: Textsize of figure labels.
    titlesize: Textsize of figure title.
    legend_anchor: Tuple of legend location on x and y in axes coordinates.
    lw: Float for the width of all plotted lines.
    path: Optional string path. Writes figure to path if set.
    """

    values = df[columns].values
    ymins, ymaxs = values.min(axis=0), values.max(axis=0)

    # Optional reverse applied to columns.
    if reverse is not None:
        for col in reverse:
            index = columns.index(col)
            ymaxs[index], ymins[index] = ymins[index], ymaxs[index]

    # Add some padding to the ranges, and recompute deltas.
    deltas = ymaxs - ymins
    ymins -= deltas * 0.1
    ymaxs += deltas * 0.1
    deltas = ymaxs - ymins

    # Prepare the figure array.
    zvalues = np.zeros_like(values)
    zvalues[:, 0] = values[:, 0]

    # Transform all data beyond the first column using broadcasting to be
    # compatible with the first axis.
    zvalues[:, 1:] = (values[:, 1:] - ymins[1:]) / deltas[1:] * deltas[
        0
    ] + ymins[0]

    # Draw the figure.
    fig, host = plt.subplots(figsize=figsize)
    axes = [host] + [host.twinx() for i in range(values.shape[1] - 1)]
    for i, ax in enumerate(axes):
        # Set the tick range manually, adapting from host.
        # Note that the actual lines will be plotted according to the
        # transformed zvalues above (i.e. all in terms of axis 0.), making
        # them essentially cosmetic axes. No lines are actually connected.
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if ax != host:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_ticks_position("right")
            # Reset drawing position of non-host axes (i fraction of len cols).
            ax.spines["right"].set_position(
                ("axes", i / (values.shape[1] - 1))
            )

    # Adjust host axis.
    host.set_xlim(0, values.shape[1] - 1)  # Remove padding.
    host.set_xticks(range(values.shape[1]))  # Set ticks before rename.
    host.set_xticklabels(list(df.columns), fontsize=labelsize)
    host.tick_params(axis="x", which="major", pad=8)  # Add vertical pad.
    host.spines["right"].set_visible(False)
    host.xaxis.tick_top()  # Move x-axis labels on top.
    host.set_title("test", fontsize=titlesize, pad=20)

    # Prepare styles of plot. Overwrite defaults with linecolors, linestyles.
    cmap = plt.get_cmap("tab10")
    colors = {idx: cmap(i) for i, idx in enumerate(df.index)}
    if linecolors is not None:
        for key in linecolors.keys():
            colors[key] = linecolors[key]
    styles = {idx: "solid" for idx in df.index}
    if linestyles is not None:
        for key in linestyles.keys():
            styles[key] = linestyles[key]

    # Plot the lines: for j submission, chart the row values by column
    for i, j in zip(df.index, range(values.shape[0])):
        host.plot(
            range(values.shape[1]),  # x
            zvalues[j, :],  # y
            c=colors[i],
            linestyle=styles[i],
            lw=lw,
        )

    host.legend(
        labels=df[columns].index,
        loc="center",
        bbox_to_anchor=legend_anchor,
        frameon=False,
    )

    if path:
        fig.savefig(
            path,
            dpi=200,
            facecolor="white",
            bbox_inches="tight",
        )
        plt.close(fig)
