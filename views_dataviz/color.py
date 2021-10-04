from typing import Any, Optional
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap


def shift_colormap(
    cmap: Colormap,
    start: float = 0,
    midpoint: float = 0.5,
    stop: float = 1.0,
    name: Optional[str] = None,
) -> Colormap:
    """Offset the center of a colormap.

    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero
    Credit: #https://gist.github.com/phobson/7916777

    ViEWS shifted rainbow: [0, 0.25, 1]

    Parameters
    ----------
    cmap: The matplotlib colormap instance to be altered
    start: Offset from lowest point in the colormap's range.
        Defaults to 0.0 (no lower ofset). Should be between
        0.0 and 1.0.
    midpoint: The new center of the colormap. Defaults to
        0.5 (no shift). Should be between 0.0 and 1.0. In
        general, this should be  1 - vmax/(vmax + abs(vmin))
        For example if your data range from -15.0 to +5.0 and
        you want the center of the colormap at 0.0, `midpoint`
        should be set to  1 - 5/(5 + 15)) or 0.75.
    stop: Offset from highets point in the colormap's range.
        Defaults to 1.0 (no upper ofset). Should be between
        0.0 and 1.0.
    name: Optional name to register cmap on.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}  # type: ignore

    # Regular index to compute the colors.
    reg_index = np.linspace(start, stop, 256)

    # Shifted index to match the data.
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 128, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    cmap = colors.LinearSegmentedColormap(name, cdict)
    if name is not None:
        plt.register_cmap(cmap=cmap, name=name)

    return cmap


def force_alpha_colormap(
    cmap: Colormap, alpha: float, name: Optional[str] = None
) -> Colormap:
    """Force alpha channel into colors of cmap.

    Adjusting the alpha column creates artefacts. To work around this,
    you can "force" the alpha into the rgb values.

    Parameters
    ----------
    cmap: The matplotlib colormap instance to be altered
    alpha: Float between 0-1 indicating alpha to apply to colormap.
    name: Optional name to register colormap on.
    """
    cmap_rgb = cmap(np.arange(256))
    for i in range(3):
        cmap_rgb[:, i] = (1 - alpha) + alpha * cmap_rgb[:, i]
    cmap = colors.ListedColormap(cmap_rgb)

    if name is not None:
        plt.register_cmap(cmap=cmap, name=name)

    return cmap
