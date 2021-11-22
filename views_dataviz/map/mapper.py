"""
Maps using matplotlib and geopandas.
"""

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import contextily as ctx

from views_dataviz import color
from views_dataviz.map import utils


class Mapper:
    """
    `Map` takes basic properties and allows the user to consecutively add
    layers to the Map object. This makes it possible to prepare mapping
    "presets" at any level of layeredness that can be built on further.

    Attributes
    ----------
    width: Integer value for width in inches.
    height: Integer value for height in inches.
    bbox: List for the bbox per [xmin, xmax, ymin, ymax].
    frame_on: Bool for whether to draw a frame around the map.
    title: Optional default title at matplotlib's default size.
    """

    def __init__(
        self,
        width,
        height,
        bbox=None,
        cmap=None,
        frame_on=True,
        title="",  # Default title without customization. (?)
    ):
        self.width = width
        self.height = height
        self.bbox = bbox  # xmin, xmax, ymin, ymax
        self.cmap = cmap
        self.fig, self.ax = plt.subplots(figsize=(self.width, self.height))
        self.texts = []
        self.ax.set_title(title)

        if frame_on:  # Remove axis ticks only.
            self.ax.tick_params(
                top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
            )
        else:
            self.ax.axis("off")

        if bbox is not None:
            self.ax.set_xlim((self.bbox[0], self.bbox[1]))
            self.ax.set_ylim((self.bbox[2], self.bbox[3]))

    def add_layer(self, gdf, cmap=None, inform_colorbar=False, **kwargs):
        """Add a geopandas plot to a new layer.

        Parameters
        ----------
        gdf: Geopandas GeoDataFrame to plot.
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
        return self

    def add_colorbar(
        self,
        cmap,
        vmin,  # TODO: make this auto by default?
        vmax,
        location="right",
        size="5%",
        pad=0.1,
        alpha=1,
        labelsize=16,
        tickparams=None,
    ):
        """Add custom colorbar to Map.

        Needed since GeoPandas legend and plot axes do not align, see:
        https://geopandas.readthedocs.io/en/latest/docs/user_guide/mapping.html

        Parameters
        ----------
        cmap: Matplotlib colormap object or string reference (e.g. "viridis").
        vmin: Minimum value of range colorbar.
        vmax: Maximum value of range colorbar.
        location: String for location of colorbar: "top", "bottom", "left"
            or "right".
        size: Size in either string percentage or number of pixels.
        pad: Float for padding between the plot's frame and colorbar.
        alpha: Float for alpha to apply to colorbar.
        labelsize: Integer value for the text size of the ticklabels.
        tickparams: Dictionary containing value-label pairs. For example:
            {0.05: "5%", 0.1: "10%"}
        """
        norm = plt.Normalize(vmin, vmax)
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        cmap = color.force_alpha_colormap(cmap=cmap, alpha=alpha)
        scalar_to_rgba = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        divider = make_axes_locatable(self.ax)
        self.cax = divider.append_axes(location, size, pad)
        self.cax.tick_params(labelsize=labelsize)
        tickvalues = (
            list(tickparams.keys()) if tickparams is not None else None
        )
        self.cbar = plt.colorbar(
            scalar_to_rgba, cax=self.cax, ticks=tickvalues
        )
        if tickparams is not None:
            self.cbar.set_ticklabels(list(tickparams.values()))
        return self

    def replace_legend_labels(self, labels):
        """Replace labels of standard mpl legend by key-value pair.

        Parameters
        ----------
        labels: Dictionary containing key-value replacements. For example:
            {0: "No conflict", 1: "Conflict"}
        """
        legend = self.ax.get_legend()
        for txt in legend.texts:
            if txt.get_text() in labels.keys():
                txt.set_text(labels[txt.get_text()])
        return self

    def add_basemap(
        self, crs=4326, source=ctx.providers.CartoDB.Voyager, **kwargs
    ):
        """Add a basemap to Map.

        Parameters
        ----------
        crs: Integer value for EPSG coordinate system used.
        source: Contextily tile to apply.
        **kwargs: `ctx.add_basemap` keyword arguments.
        """
        ctx.add_basemap(self.ax, crs=crs, source=source, **kwargs)
        return self

    def add_text(self, gdf, column, **kwargs):
        """Add text by row value and geom.centroid.

        Texts collected into self.texts can be used for ex-post adjustments,
        for instance https://github.com/Phlya/adjustText.

        Parameters
        ----------
        gdf: Geopandas GeoDataFrame to plot.
        column: Text column in gdf to apply.
        """
        geom = gdf._geometry_column_name
        for _, row in gdf.iterrows():
            self.texts.append(
                self.ax.annotate(
                    text=row[column], xy=row[geom].centroid.coords[0], **kwargs
                )
            )
        return self

    def add_title(self, title, **kwargs):
        """Add a custom title. Replaces default.

        Parameters
        ----------
        title: String title.
        **kwargs: Matplotlib `ax.set_title` keyword arguments.
        """
        self.ax.set_title(title, **kwargs)
        return self

    def add_views_textbox(self, text, textsize=15):
        """Add ViEWS textbox to figure. Logo and url are hardcoded.

        Parameters
        ----------
        text: String text. Note that newlines can be used, for example:
            "Model A\nr_2021_12_01"
        textsize: Integer for textsize.
        """
        utils.add_textbox_to_ax(self.fig, self.ax, text, textsize)
        return self

    def save(
        self, path, dpi=200, **kwargs
    ):  # Just some defaults to reduce work.
        """Save Map figure to file.

        Parameters
        ----------
        path: String path, e.g. "./example.png".
        dpi: Integer dots per inch. Increase for higher resolution figures.
        **kwargs: Matplotlib `savefig` keyword arguments.
        """
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
        plt.close(self.fig)
