"""Continuum plot"""

from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from adjustText import adjust_text
from views_dataviz.color import get_rgb_from_continuum


def plot_continuum(
    continuum_dict: Dict[str, int],
    title: str = "",
    titlesize: float = 16.0,
    labelsize: float = 12.0,
    ticksize: float = 12.0,
    figsize: Tuple[float, float] = (14, 5),
    adjust_kwargs: Dict[str, Any] = None,
    path: str = None,
):
    """Plots annotations on a 0-100 continuum.

    Uses adjustText which can often be imperfect. You may have
        to tinker with its parameters.

    Parameters
    ----------
    continuum_dict: Dictionary containing label: position pairs.
    title: Optional title to give to the figure.
    titlesize: Float for the textsize of the title.
    labelsize: Float for the textsize of the labels.
    ticksize: Float for the textsize of the tick labels.
    figsize: Tuple of figure size (width, height) in inches.
    adjust_kwargs: Dict of keyword arguments for adjustText.
    path: Optional. Path to write figure to.
    """
    _, ax = plt.subplots(figsize=figsize)
    plt.title(title, loc="left", fontsize=titlesize)

    # Empty list to store text annotations; used for adjust_text().
    texts = []

    # Location list.
    xy_main = []

    # Plot markers and text annotations.
    for name, location in continuum_dict.items():
        if (location, 0) in xy_main:
            xy_sub = [i for i in xy_main if i[0] == location]
            y = max(xy_sub, key=lambda x: x[1])[1]
            y = y + 1
        else:
            y = 0

        # plot with wiggle to avoid sticking behavior (not 100%).
        texts.append(
            plt.text(
                location + np.random.random() / 1000,
                y + np.random.random() / 1000,
                name,
                fontsize=labelsize,
            )
        )
        plt.plot(
            location,
            y,
            marker="o",
            markersize=25,
            color=get_rgb_from_continuum(location),
            alpha=0.5,
        )
        xy_main.append((location, y))

    # Set x-axis to middle of figure and hide other axes.
    ax.spines["left"].set_position("center")
    ax.spines["left"].set_color("none")
    # ax.spines["left"].set_smart_bounds(True)
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_position("center")
    ax.spines["bottom"].set_color("#505050")
    ax.spines["top"].set_color("none")

    # Set ax properties (lims, ticks, thickness).
    ax.set_xlim(-2.5, 102.5)
    plt.setp(ax.spines.values(), linewidth=4)
    ax.set_xticks(list(range(0, 105, 5)))
    ax.tick_params(axis="x", which="major", labelsize=ticksize)
    ax.set_ylim(-5, 5)
    ax.tick_params(length=5)
    ax.set_yticks([])

    # Create virtual space around axis and ticks to repel against.
    patch = patches.Rectangle(
        (-2.5, -0.6), 105, 1.5, fill=True, alpha=0
    )
    ax.add_patch(patch)

    # Repel function, with arrow properties set inside.
    default_kwargs = {
        "autoalign": False,
        "ha": "left",
        "va": "center",
        "add_objects": [patch],
        "force_objects": (0, 0.5),
        "only_move": {"points": "x", "text": "y", "objects": "y"},
        "expand_text": (1, 1.5),
        "force_text": (0.1, 1),
        "expand_points": (1.2, 1),
        "force_points": (0.3, 0),
        "arrowprops": dict(arrowstyle="-|>", color="black", lw=0.7),
    }
    if adjust_kwargs is not None:
        kwargs = adjust_kwargs
    else:
        kwargs = default_kwargs
    adjust_text(texts, **kwargs)

    if path is not None:
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close("all")
