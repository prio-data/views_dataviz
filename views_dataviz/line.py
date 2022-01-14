"""
Line plotter
"""

from typing import Tuple, List
import matplotlib.pyplot as plt


class Line:
    """
    Line wraps around matplotlib's `plot`, cutting away some of the boilerplate
    that is usually required when making a lineplot.

    Attributes
    ----------
    cmap: String for a matplotlib colormap.
    figsize: Tuple of width and height of figure in inches.
    lw: Linewidth to apply to plot. You can override this by line by calling
        linewidth in the add_line method.
    bbox_to_anchor: Tuple of x, y determining the location of the legend in
        axes coordinates.
    hide_spines: Spines to hide. Removes the top and right spines by default.
    labelsize: Textsize of the ylabel(s).
    ticksize: Textsize of the ticks. Assumes same size for x and y. If this
        is not desirable, take the ax from the Line object and set manually.
    """

    def __init__(
        self,
        cmap: str = "tab20",
        figsize: Tuple[int, int] = (7, 5),
        lw: int = 3,
        bbox_to_anchor: Tuple[float, float] = (1.2, 1),
        hide_spines: List[str] = ["top", "right"],
        labelsize: int = 14,
        ticksize: int = 12,
    ):
        self.cmap = cmap
        self.lw = lw
        self.bbox_to_anchor = bbox_to_anchor
        self.hide_spines = hide_spines
        self.labelsize = labelsize
        self.ticksize = ticksize
        self.fig, self.ax1 = plt.subplots(figsize=figsize)
        self.ax1.grid(b=True, axis="y", color="lightgrey", linestyle="-")
        self.ax2 = None
        self.ax_index = 0

    def __drop_spines(self, ax):
        for spine in self.hide_spines:
            ax.spines[spine].set_visible(False)

    def set_legend(self):
        """Add legend."""
        if self.ax2 is not None:
            h1, l1 = self.ax1.get_legend_handles_labels()
            h2, l2 = self.ax2.get_legend_handles_labels()
            self.ax1.legend(
                h1 + h2,
                l1 + l2,
                loc="best",
                bbox_to_anchor=self.bbox_to_anchor,
                frameon=False,
            )
        else:
            self.ax1.legend(
                loc="best", bbox_to_anchor=self.bbox_to_anchor, frameon=False
            )
        return self

    def add_line(self, *args, twinx=False, ylabel="", ylabelpad=10, **kwargs):
        """Add a line layer to the figure frame.

        Parameters
        ----------
        *args: Positional arguments that go into the matplotlib plot call.
        twinx: Bool indicating whether to use the right axis for the scale of
            the line, thereby making a secondary axis.
        ylabel: String for the ylabel of the particular ax.
        ylabelpad: Margin for the ylabel.
        **kwargs: Keyword arguments that go into the matplotlib plot call.
        """
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = self.lw
        # If color not explicitly set, take index color from cmap.
        if "c" not in kwargs and "color" not in kwargs:
            cmap = plt.get_cmap(self.cmap)
            kwargs["color"] = cmap(self.ax_index)
        if twinx:
            self.ax2 = self.ax1.twinx()
            self.ax2.plot(*args, **kwargs)
            if self.hide_spines is not None:
                self.__drop_spines(self.ax1)
                self.__drop_spines(self.ax2)
            self.ax2.set_ylabel(
                ylabel, size=self.labelsize, labelpad=ylabelpad
            )
            self.ax2.tick_params(
                axis="both", which="major", labelsize=self.ticksize
            )
        else:
            self.ax1.plot(*args, **kwargs)
            if self.hide_spines is not None:
                self.__drop_spines(self.ax1)
            self.ax1.set_ylabel(
                ylabel, size=self.labelsize, labelpad=ylabelpad
            )
            self.ax1.tick_params(
                axis="both", which="major", labelsize=self.ticksize
            )

        self.ax_index += 1
        return self

    def save(self, path, dpi=200, **kwargs):
        """Save figure to file.

        Parameters
        ----------
        path: String path, e.g. "./example.png".
        dpi: Integer dots per inch. Increase for higher resolution figures.
        **kwargs: Matplotlib `savefig` keyword arguments.
        """
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
        plt.close(self.fig)
