"""
Mapping utilities
"""

from typing import Any, Dict, List, Tuple, Union
import os
import numpy as np

from matplotlib import pyplot as plt, image as mpimg, colors  # type: ignore
from matplotlib.offsetbox import AnchoredText, OffsetImage, AnnotationBbox  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore


def odds_to_prob(odds: Any) -> Any:
    """Cast odds ratio to probability."""
    return odds / (odds + 1)


def logodds_to_prob(logodds: Any) -> Any:
    """Cast logodds to probability."""
    return odds_to_prob(np.exp(logodds))


def prob_to_logodds(p):
    """Cast probability to log-odds."""
    return np.log(prob_to_odds(p))


def prob_to_odds(p, clip=True):
    """Cast probability into odds."""

    if isinstance(p, list):
        p = np.array(p)

    if clip:
        offset = 1e-10
        offset = 1e-10
        upper = 1 - offset
        lower = 0 + offset
        p = np.clip(p, lower, upper)

    # Check for probs greq 1 because odds of 1 is inf which might break things.
    if np.any(p >= 1):
        msg = "probs >= 1 passed to get_odds, expect infs"
        warnings.warn(msg)

    odds = p / (1 - p)
    return odds


def make_ticks(var_scale: str) -> Dict[str, Any]:
    """Make tick dictionary with 'values' and 'labels' depending on var_scale.

    Args:
        var_scale: "logodds" or "prob".
    Returns:
        ticks: dict with 'values' and 'labels'.
    """

    def format_prob_to_pct(p: float) -> str:
        """Cast probabilities to pct (%) formatted strings."""

        if not 0 <= p <= 1:
            raise RuntimeError("Value does not look like a probability.")

        pct = p * 100
        if pct == int(pct):
            pct = int(pct)

        return f"{pct}%"

    def make_ticks_logit() -> Tuple[List[float], List[str]]:
        """ Make logistic ticks """
        ticks_logit = []
        ticks_strings = []
        ticks = [
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.4,
            0.6,
            0.8,
            0.9,
            0.95,
            0.99,
        ]

        for tick in ticks:
            ticks_logit.append(prob_to_logodds(tick))
            ticks_strings.append(format_prob_to_pct(tick))

        # Make the lower than/equal to for 0.001.
        ticks_strings[0] = "<= " + ticks_strings[0]

        return ticks_logit, ticks_strings

    def make_ticks_probs() -> Tuple[List[float], List[str]]:
        ticks_strings = []
        ticks_probs = [
            0.001,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.99,
        ]

        for tick in ticks_probs:
            ticks_strings.append(str(tick))

        return ticks_probs, ticks_strings

    if var_scale == "logodds":
        values, labels = make_ticks_logit()
    elif var_scale == "prob":
        values, labels = make_ticks_probs()

    return dict(zip(values, labels))


def add_textbox_to_ax(
    fig: Any,
    ax: Any,
    text: str,
    textsize: int,
    corner: str = "lower left",
    corner_offset: float = 0.3,
    textbox_pad: float = 0.4,
) -> None:
    """Add bounded white textbox to ax, with url and logo.

    Args:
        fig: Matplotlib figure.
        ax: Matplotlib ax.
        text: Text to draw in box.
        textsize: Font size.
        corner: Location of anchored textbox.
        corner_offset: How far from each corner to draw box. Note the
            other elements (url and logo) depend on this.
        textbox_pad: Padding inside the main textbox.
    """
    # Set anchored textbox.
    text_anchor = AnchoredText(
        text,
        loc=corner,
        pad=textbox_pad,
        borderpad=corner_offset,
        prop={"fontsize": textsize - 5},
    )
    text_anchor.patch.set(alpha=0.8)
    ax.add_artist(text_anchor)

    # Once textbox is drawn up, get bbox coordinates for that.
    fig.canvas.draw()
    renderer = fig.canvas.renderer
    coords = ax.transData.inverted().transform(
        text_anchor.get_window_extent(renderer)
    )

    # Params depending on selected corner. Use the coords to inset the url.
    # [0][0]: xmin, [0][1]: ymin, [1][0]: xmax, [1][1]: ymax.
    cornerparams = {
        "lower left": {
            "xy": (coords[0][0], coords[1][1]),
            "offset": (0, 2),
            "ha": "left",
            "va": "bottom",
        },
        "lower right": {
            "xy": (coords[1][0], coords[1][1]),
            "offset": (0, 2),
            "ha": "right",
            "va": "bottom",
        },
    }

    if corner not in cornerparams:
        raise KeyError(f"{corner} is not a valid corner (yet).")

    style = dict(
        facecolor="white",
        alpha=0,
        edgecolor="red",  # For testing.
        boxstyle="square, pad=0",
    )
    text_url = ax.annotate(
        "http://views.pcr.uu.se",
        xy=cornerparams[corner]["xy"],
        xytext=cornerparams[corner]["offset"],
        textcoords="offset points",
        fontsize=textsize - 5,
        bbox=style,
        ha=cornerparams[corner]["ha"],
        va=cornerparams[corner]["va"],
    )

    # Again once textbox is drawn up, get bbox coordinates for that.
    fig.canvas.draw()
    text_bbox = text_url.get_bbox_patch()
    text_bbox = text_bbox.get_extents()
    urlcoords = ax.transData.inverted().transform(text_bbox)

    # Add the ViEWS logo.
    this_dir = os.path.dirname(__file__)
    path_logo_views = os.path.join(this_dir, "assets/logo_transparent.png")
    logo_views = mpimg.imread(path_logo_views)

    # Define a 1st position to annotate.
    xy = (urlcoords[0][0], urlcoords[1][1])
    imagebox = OffsetImage(logo_views, zoom=0.2)
    imagebox.image.axes = ax

    ab = AnnotationBbox(
        imagebox,
        xy,
        xycoords="data",
        xybox=(55, 20),  # Arbitrary offset in points that looks ok.
        frameon=False,
        boxcoords="offset points",
    )
    ax.add_artist(ab)
