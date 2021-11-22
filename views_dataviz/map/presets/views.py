""" 
ViEWS mapping presets.
"""

import numpy as np
from views_dataviz import color
import matplotlib.pyplot as plt
from views_dataviz.map import Mapper, utils


BBOX_AFRICA = [-18.5, 52.0, -35.5, 38.0]
BBOX_AME = [-18.5, 64.0, -35.5, 43.0]  # (xmin, xmax, ymin, ymax)
CMAPS = {
    "prob": plt.get_cmap("rainbow"),
    "logodds": color.shift_colormap(plt.get_cmap("rainbow")),
    "delta": plt.get_cmap("seismic"),
}
TICKPARAMS = {
    "prob": utils.make_ticks("prob"),
    "logodds": utils.make_ticks("logodds"),
    "delta": dict(
        zip(
            np.round(np.arange(-1, 1.2, 0.2), 1),
            [str(i) for i in np.round(np.arange(-1, 1.2, 0.2), 1)],
        )
    ),
}


class ViewsMap(Mapper):
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
        bbox=None,
        cmap="viridis",
        frame_on=True,
        title="",
        label="",
        scale=None,
        vmin=None,
        vmax=None,
        tickparams=None,
    ):
        if isinstance(bbox, str):
            bbox_string = bbox.lower()
            if bbox_string not in ("africa", "africa_middle_east"):
                raise RuntimeError(
                    f"{bbox_string} is not among options 'africa' or 'africa_middle_east'."
                )
            if bbox_string == "africa":
                bbox = BBOX_AFRICA
            if bbox_string == "africa_middle_east":
                bbox = BBOX_AME
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
            self.cmap = plt.get_cmap(self.cmap)
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
            tickparams=self.tickparams,
        )
        self.n_textbox = 0

    def add_layer(self, gdf, cmap=None, inform_colorbar=False, **kwargs):
        """Add a geopandas plot to a new layer.

        Overrides method in parent, adding the default views textbox
        after plotting an initial layer.

        Parameters
        ----------
        gdf: Geopandas GeoDataFrame to plot.
        cmap: Optional matplotlib colormap object or string reference
            (e.g. "viridis").
        inform_colorbar: Set or overwrite colorbar with the current layer.
        **kwargs: Geopandas `.plot` keyword arguments.
        """
        colormap = self.cmap if cmap is None else cmap
        # If inform_colorbar, replace cax if exists and set with vmin, vmax.
        if inform_colorbar and "column" in kwargs:
            if hasattr(self, "cax"):
                self.cax.remove()
            if "vmin" not in kwargs:
                self.vmin = gdf[kwargs["column"]].min()
            else:
                self.vmin = kwargs["vmin"]
            if "vmax" not in kwargs:
                self.vmax = gdf[kwargs["column"]].max()
            else:
                self.vmax = kwargs["vmax"]
            Mapper.add_colorbar(self, colormap, self.vmin, self.vmax)
        self.ax = gdf.plot(ax=self.ax, cmap=colormap, **kwargs)
        if self.n_textbox == 0:
            Mapper.add_views_textbox(self, text=self.label, textsize=16)
            self.n_textbox += 1
        return self
