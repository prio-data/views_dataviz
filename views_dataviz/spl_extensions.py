"""Sepplotlib extensions for ViEWS"""

from matplotlib.patches import ConnectionPatch
import sepplotlib as spl

from views_dataviz.map.presets import ViewsMap


class ModelCriticismMap(spl.ModelCriticismPlot):
    """
    Extends spl.ModelCriticismPlot with a mapping method.
    Overrides original labelling method.

    Observations are matched with *centroid* locations in the gdf
    by the polygon identifier provided in `lab`. Make sure that this
    column is the same in df as in your gdf (e.g. "pg_id").

    Attributes
    ----------
    df: pd.DataFrame containing the predictions, actuals and labels.
    y_true: Column name for the actual observations.
    y_pred: Column name for the model predictions.
    lab: Column name for the row labels.
    mc_kwargs: Dictionary of keyword arguments used by ModelCriticismPlot.
    map_kwargs: Dictionary of keyword arguments used by ViewsMap.
    title: Optional title.
    titlesize: Textsize of title.
    """

    def __init__(
        self,
        df,
        y_true,
        y_pred,
        lab,
        mc_kwargs,
        map_kwargs,
        title="",
        titlesize=18,
    ):
        super().__init__(df, y_true, y_pred, lab, **mc_kwargs)
        self.set_map_ax()
        self.mapper = ViewsMap(figure=(self.fig, self.map_ax), **map_kwargs)
        self.fig.suptitle(title, x=0, size=titlesize)

    def annotate(self):
        """Suppresses the annotation method in the parent class."""
        return self

    def set_map_ax(self):
        """Add an ax for the map."""
        self.map_ax = self.fig.add_axes([1.1, 0, 1.1, 1])
        self.map_ax.tick_params(
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
        )
        for axis in ["top", "bottom", "left", "right"]:
            self.map_ax.spines[axis].set_linewidth(self.framesize)

    def plot_map_layer(self, suppress_colorbar=True, **kwargs):
        """Plots a map layer."""
        self.mapper.add_layer(**kwargs)
        if suppress_colorbar:
            if hasattr(self.mapper, "cax"):
                self.mapper.cax.remove()  # Remove default colorbar.
        return self

    def connect(self, gdf):
        """
        Finds the appropriate map coordinate by matching labs.
        """
        for idx, row in self.df.loc[self.df.worst_fp == 1].iterrows():
            # For some reason geopandas loses some information when queried.
            mapindex = gdf.loc[gdf[self.lab] == row[self.lab]].index[0]
            mapcoord = gdf.loc[mapindex].geom.centroid.coords[0]
            con = ConnectionPatch(
                xyA=(1 + self.pad, idx),
                xyB=mapcoord,
                coordsA="data",
                coordsB="data",
                axesA=self.axs[0],
                axesB=self.map_ax,
                color=row["fgcolor"],
            )
            self.map_ax.add_artist(con)
        for idx, row in self.df.loc[self.df.worst_fn == 1].iterrows():
            mapindex = gdf.loc[gdf[self.lab] == row[self.lab]].index[0]
            mapcoord = gdf.loc[mapindex].geom.centroid.coords[0]
            con = ConnectionPatch(
                xyA=(1 + self.pad, idx),
                xyB=mapcoord,
                coordsA="data",
                coordsB="data",
                axesA=self.axs[0],
                axesB=self.map_ax,
                color=row["fgcolor"],
            )
            self.map_ax.add_artist(con)
        return self
