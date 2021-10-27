""" 
ViEWS mapping presets.
"""

import numpy as np
from views_dataviz import color
import matplotlib.pyplot as plt
from views_dataviz.map import Mapper, utils


BBOX_AFRICA = [-18.5, 52.0, -35.5, 38.0]
CMAPS = {
    "prob": plt.get_cmap("rainbow"),
    "logodds": color.shift_colormap(plt.get_cmap("rainbow")),
    "delta": plt.get_cmap("seismic")
}
TICKPARAMS = {
    "prob": utils.make_ticks("prob"),
    "logodds": utils.make_ticks("logodds"),
    "delta": dict(
        zip(
            np.round(np.arange(-1, 1.2, 0.2), 1), 
            [str(i) for i in np.round(np.arange(-1, 1.2, 0.2), 1)]
        )   
    )
}


class ViewsAfrica(Mapper):
    """
    Inherits from Mapper. Runs methods on init that build the ViEWS defaults.

    Attributes
    ----------
    title: Add a custom str title.
    label: Custom str label to add to textbox.
    scale: Scale to set map to. Either "logodds", "prob", "delta", or None.
    vmin: Minimum value of scale.
    vmax: Maximum value of scale.
    tickparams: Dictionary of custom tick parameters, by key-value pairs. For
        example: {0.05: "5%", 0.1: "10%"}.
    """

    def __init__(
        self, 
        width=10,
        height=10, 
        bbox=BBOX_AFRICA,
        cmap="viridis",
        frame_on=True, 
        title="",
        label="",
        scale=None,
        vmin=None,
        vmax=None,
        tickparams=None,
    ):
        super().__init__(width, height, bbox, cmap, frame_on, title)
        self.label = label
        if scale not in (None, "prob", "logodds", "delta"):
            raise ValueError(
                "Invalid scale. Options: 'prob', 'logodds', 'delta', or None."
            )
        self.scale = scale
        if self.scale is not None:
            self.cmap = CMAPS[scale]
            self.vmin = min(TICKPARAMS[scale].keys())
            self.vmax = max(TICKPARAMS[scale].keys())
            self.tickparams = TICKPARAMS[scale]
        else:
            self.cmap = plt.get_cmap(self.cmap)  # TODO: fix.
            self.vmin = vmin
            self.vmax = vmax
            self.tickparams = tickparams
        Mapper.add_title(self, title=title, size=25)
        Mapper.add_colorbar(
            self,
            cmap=self.cmap, 
            vmin=self.vmin,
            vmax=self.vmax,
            pad=0.1, 
            labelsize=16,
            tickparams=self.tickparams
        )
        Mapper.add_views_textbox(
            self,
            text=label, 
            textsize=16
        )
