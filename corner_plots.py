#!/usr/bin/env python3
"""
Multi-run corner plots for ForMoSA posteriors.

Public entry point: plot_corner_from_config_paths().

Overview
--------
This module compares several ForMoSA nested-sampling runs on a single corner
plot. It handles data loading, posterior preparation, range computation, and
all figure decoration (reference markers, result texts, legend).

Typical usage::

    from corner_plots import plot_corner_from_config_paths, CornerPlotOptions, CornerPlotStyle

    fig = plot_corner_from_config_paths(
        config_paths=["run1.ini", "run2.ini", "run3.ini"],
        labels=["Run 1", "Run 2", "Run 3"],
        values={"Teff": 1800, "log(g)": 4.0, "C/O": 0.55},
        mode="concatenated",
        plot_options=CornerPlotOptions(corner_bins=60, zoom=True),
        style=CornerPlotStyle(figsize=(12, 16)),
    )
    fig.savefig("corner.pdf")
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import corner
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

from ForMoSA.config.global_config import ConfigLoader
from ForMoSA.analysis import Analysis


# ============================================================
# LATEX LABELS
# ============================================================

LATEX_LABELS = {
    "Teff":   r"T_{\rm eff}",
    "log(g)": r"\log(g)",
    "[M/H]":  r"[{\rm M/H}]",
    "C/O":    r"{\rm C/O}",
    "fsed":   r"f_{\rm sed}",
    "f_sed":  r"f_{\rm sed}",
    "rv":     r"RV",
    "vsini":  r"v \, \sin i",
}

BOLD_LATEX_LABELS = {
    "Teff":   r"\mathbf{T}_{\mathbf{eff}}",
    "log(g)": r"\mathbf{log(g)}",
    "[M/H]":  r"\mathbf{[M/H]}",
    "C/O":    r"\mathbf{C/O}",
    "fsed":   r"\mathbf{f}_{\mathbf{sed}}",
    "f_sed":  r"\mathbf{f}_{\mathbf{sed}}",
    "rv":     r"\mathbf{RV}",
    "vsini":  r"\mathbf{v\,sin\,i}",
}


# ============================================================
# CONFIGURATION DATACLASSES
# ============================================================

@dataclass
class CornerPlotOptions:
    """
    Switches and parameters that control *what* is plotted.

    These options determine the number of histogram bins, smoothing behaviour,
    which runs are highlighted, and which parameters are included. They map
    mostly to arguments forwarded to ``corner.corner()``.

    Attributes
    ----------
    normalized : bool
        If True, each 1D diagonal histogram is rescaled so its peak equals 1.
        Useful for comparing posteriors with very different sample counts.
    zoom : bool
        If True, draw zoomed inset histograms above each diagonal panel. The widest
        run is excluded from the inset range. Can be combined with any ``mode``.
        Requires at least two runs.
    corner_bins : int
        Number of bins used in the 2D off-diagonal panels.
    hist_bins : int
        Number of bins used in the 1D diagonal panels (zoom insets and concatenated overlay).
    n_highlighted_runs : int
        In ``concatenated_full`` mode, only the first *n_highlighted_runs*
        runs are drawn in colour; the rest are gray. Has no effect in
        ``standard`` or ``concatenated`` modes.
    selected_params : list[int] or None
        Indices into the common parameter list to restrict the corner plot to a
        subset of parameters (e.g. ``[1, 2]`` keeps only the 2nd and 3rd
        parameters). ``None`` (default) plots every common parameter.
    smooth_2d : float, list[float], or None
        2D smoothing passed to ``corner.corner()``. A scalar applies the same
        value to every run; a list must have one entry per run. ``None``
        disables smoothing.
    plot_datapoints : bool
        Whether to scatter the raw posterior samples in 2D panels.
    plot_2d : bool
        Whether to draw filled contours and a density map in 2D panels.
        When False, only the contour lines are shown.
    levels : list[float]
        Probability levels for the 2D contours.
"""
    normalized: bool = False
    zoom: bool = False

    corner_bins: int = 80
    hist_bins: int = 30

    n_highlighted_runs: int = 3
    
    selected_params: list[int] | None = None
    
    smooth_2d: float | list[float] | None = 1
    plot_datapoints: bool = False
    plot_2d: bool = True
    levels: list[float] = field(default_factory=lambda: [0.3935, 0.8647, 0.9889])


@dataclass
class CornerPlotStyle:
    """
    All visual and layout properties of the corner figure.

    Separating style from plot logic makes it straightforward to produce
    publication-ready variants by passing a customised ``CornerPlotStyle``
    without touching the plotting code.

    Attributes
    ----------
    figsize : tuple[float, float]
        Overall figure size in inches ``(width, height)``.
    result_fontsize : float
        Font size for the per-panel result texts (q50 ± 1σ summaries).
    label_size : float
        Font size for the parameter-name axis labels.
    tick_label_size : float
        Font size for the numeric tick labels on every axis.
    legend_fontsize : float
        Font size for the legend entries.
    result_base : float
        Vertical position of the *lowest* result text line, in transAxes
        units relative to the diagonal panel. Values above 1 place the text
        above the panel top edge.
    result_step : float
        Vertical gap between consecutive result text lines, as a fraction of
        the diagonal panel height. Automatically converted to figure
        coordinates so spacing stays consistent regardless of how many
        parameters are plotted.
    bottom : float
        Bottom boundary of the subplot grid (figure fraction), including the
        extra margin for the manually placed x-axis labels.
    hspace : float
        Vertical spacing between subplot rows (fraction of average axes height).
    grid_top : float
        Starting value for the top boundary of the subplot grid, reduced
        automatically as the number of text lines grows.
    top_shrink_per_line : float
        Rate at which the top boundary is reduced per extra text line beyond 2.
    x_labelpad : float
        Additional downward offset (transAxes) for the manual x-axis labels.
    y_labelpad : float
        Horizontal offset (transAxes) of the y-axis labels from the left
        edge of the panel. Negative values move the label further left.
    legend_loc : str
        ``loc`` argument passed to ``fig.legend()``.
    legend_bbox : tuple[float, float]
        Base anchor position ``(x, y)`` for the legend in figure fractions.
        The ``y`` component is automatically shifted by the difference between
        the actual subplot top and ``grid_top``, so the legend follows the
        text block up or down across modes without manual adjustment.
    zoom_y0 : float
        Vertical origin of the inset zoom histogram, in transAxes units.
    zoom_height : float
        Height of the inset zoom histogram, in transAxes units.
    zoom_text_gap : float
        Extra gap between the zoom inset top and the first text line.
    zoom_grid_top : float
        Equivalent of *grid_top* used when ``zoom=True`` in ``CornerPlotOptions``.
    zoom_reference_linewidth : float
        Linewidth of the reference vertical line drawn inside the zoom inset.
    zoom_connection_linewidth : float
        Linewidth of the vertical guide lines connecting the zoom inset to the
        full-range diagonal panel.
    margin_fraction : float
        Fractional margin added on both sides of any computed range (both the
        global axis range and the zoom inset range).
    range_percentiles : tuple[float, float]
        Robust percentile bounds used when computing axis and zoom ranges,
        before the margin and any reference value are applied.
    plot_order_percentiles : tuple[float, float]
        Percentiles used to rank runs by posterior width (wider runs are drawn
        first so narrower runs appear on top).
    colors_by_run : list[str] or None
        One Matplotlib colour per run. If ``None``, the default colour cycle
        is used.
    actual_values_color : str
        Colour for reference markers (true / injected values).
    combined_color : str
        Colour for the concatenated / combined posterior overlay.
    hist_linewidth : float
        Linewidth for individual run 1D histograms.
    concatenated_hist_linewidth : float
        Linewidth for the concatenated posterior 1D histogram.
    contour_linewidth : float
        Linewidth for 2D contour lines.
    reference_linewidth : float
        Linewidth for reference vertical/horizontal lines and scatter markers.
    reference_marker_size : float
        Marker size (``s`` argument to ``scatter``) for reference points in 2D
        panels.
    reference_marker_edgecolor : str
        Edge colour of the reference scatter marker.
    legend_linewidth : float
        Linewidth of the colour swatches drawn in the legend.
    """
    # Figure and fonts.
    figsize: tuple[float, float] = (10.0, 14.0)
    result_fontsize: float = 13
    label_size: float = 17
    tick_label_size: float = 12
    legend_fontsize: float = 15

    # Subplot layout.
    result_base: float = 1.14
    result_step: float = 0.27
    bottom: float = 0.32
    hspace: float = 0.08
    grid_top: float = 0.88
    top_shrink_per_line: float = 0.025
    x_labelpad: float = 0.25
    y_labelpad: float = -0.1

    # Legend placement.
    legend_loc: str = "upper right"
    legend_bbox: tuple[float, float] = (0.95, 0.95)

    # Zoom-mode layout.
    zoom_y0: float = 1.09
    zoom_height: float = 1.00
    zoom_text_gap: float = 0.18
    zoom_grid_top: float = 0.78
    zoom_reference_linewidth: float = 1.0
    zoom_connection_linewidth: float = 1.5

    # Axis ranges.
    margin_fraction: float = 0.08
    range_percentiles: tuple[float, float] = (0.5, 99.5)
    plot_order_percentiles: tuple[float, float] = (16, 84)

    # Colours.
    colors_by_run: list[str] | None = None
    actual_values_color: str = "black"
    combined_color: str = "0.2"

    # Line and marker widths.
    hist_linewidth: float = 1.0
    concatenated_hist_linewidth: float = 1.5
    contour_linewidth: float = 0.5
    reference_linewidth: float = 2.0
    reference_marker_size: float = 80.0
    reference_marker_edgecolor: str = "white"
    legend_linewidth: float = 2.2


@dataclass
class CornerData:
    """
    Posterior data container produced by the preparation pipeline and consumed
    by ``CornerPlotter``. Not normally constructed by hand.

    Attributes
    ----------
    parameters : list[str]
    reference_values : list[float or None]
    samples_by_run : list[np.ndarray], shape (n_samples, n_parameters) per run
    weights_by_run : list[np.ndarray], shape (n_samples,) per run
    labels_by_run : list[str]
    colors_by_run : list[str]
    ranges : list[tuple[float, float]], (xmin, xmax) per parameter
    plot_order : list[int]
        Run indices from widest to narrowest posterior.
    """
    parameters: list[str]
    reference_values: list[float | None]
    samples_by_run: list[np.ndarray]
    weights_by_run: list[np.ndarray]
    labels_by_run: list[str]
    colors_by_run: list[str]
    ranges: list[tuple[float, float]]
    plot_order: list[int]

    @property
    def n_runs(self):
        """Number of runs."""
        return len(self.samples_by_run)

    @property
    def ndim(self):
        """Number of plotted parameters."""
        return len(self.parameters)


# ============================================================
# MAIN PLOTTER
# ============================================================

@dataclass
class CornerPlotter:
    """
    Draw a multi-run corner plot from prepared posterior data.

    This class is not normally instantiated directly. Use the public function
    ``plot_corner_from_config_paths()`` instead, which handles data loading
    and calls this class internally.

    Parameters
    ----------
    data : CornerData
        Posterior samples, weights, labels, colours, and axis ranges for all
        runs, as produced by the data-preparation pipeline.
    values : dict or None
        Reference values dict (same object forwarded from the public function),
        used only to decide whether to add the "Actual values" legend entry.
    mode : str
        One of the valid mode strings, controlling which diagonal
        histogram overlay is drawn and which runs are highlighted.
    plot_options : CornerPlotOptions
        Switches and bin/smoothing parameters.
    style : CornerPlotStyle
        All visual and layout properties.

    Notes
    -----
    ``plot()`` is the only public method. All other methods are internal and
    follow a ``_verb_noun`` naming convention.
    """
    data: CornerData
    values: dict | None = None
    mode: str = "standard"
    plot_options: CornerPlotOptions = field(default_factory=CornerPlotOptions)
    style: CornerPlotStyle = field(default_factory=CornerPlotStyle)

    fig: Figure | None = field(init=False, default=None)
    axes: np.ndarray | None = field(init=False, default=None)
    layout: dict | None = field(init=False, default=None)

    # ---- semantic mode helpers ----------------------------------------

    @property
    def _is_full_mode(self):
        """True when secondary runs are grayed out (concatenated_full mode)."""
        return self.mode == "concatenated_full"

    @property
    def _show_combined(self):
        """True when an all-runs concatenated posterior overlay is drawn."""
        return self.mode in {"concatenated_full", "concatenated"}

    # ---- run colours and draw ordering --------------------------------

    def run_color(self, run_index):
        """Return the colour for one run; gray for secondary runs in full mode."""
        if self._is_full_mode and run_index >= self.plot_options.n_highlighted_runs:
            return "gray"
        return self.data.colors_by_run[run_index]

    @property
    def visible_run_indices(self):
        """Indices of runs shown in the legend and result texts."""
        if self._is_full_mode:
            return list(range(min(self.plot_options.n_highlighted_runs, self.data.n_runs)))
        return list(range(self.data.n_runs))

    @property
    def highlighted_order(self):
        """Visible runs only, in plot order (narrowest last -> on top)."""
        visible = set(self.visible_run_indices)
        return [i for i in self.data.plot_order if i in visible]

    @property
    def n_text_lines(self):
        """Number of result-text lines above each diagonal panel."""
        if self.mode == "concatenated":
            return 1
        return len(self.visible_run_indices) + (1 if self._is_full_mode else 0)

    @cached_property
    def _all_runs(self):
        """Concatenated samples and weights of every run (computed once)."""
        return (np.concatenate(self.data.samples_by_run),
                np.concatenate(self.data.weights_by_run))

    # ---- main entry point ---------------------------------------------

    def plot(self) -> Figure:
        """
        Draw and decorate the corner figure.

        Returns
        -------
        Figure
            The completed Matplotlib figure.
        """
        smooth_2d = self.plot_options.smooth_2d
        if smooth_2d is not None:
            if np.isscalar(smooth_2d):
                self.plot_options.smooth_2d = [smooth_2d] * self.data.n_runs
            elif len(smooth_2d) != self.data.n_runs:
                raise ValueError("smooth_2d must be None, a scalar, or one value per run.")
        self.layout = self._get_corner_layout()
        self.fig, self.axes = self._draw_base_corner()

        if self.plot_options.normalized:
            self._normalize_diagonal_histograms()
        if self._show_combined:
            self._draw_concatenated_histograms()
        if self.plot_options.zoom:
            self._draw_zoom_histograms()

        self._add_result_texts()
        self._add_reference_markers()
        self._add_legend()
        self._format_axes()

        return self.fig


    def _get_corner_layout(self):
        """Compute figure-level layout geometry.

        Returns
        -------
        dict
            Keys: ``zoom_bounds``, ``result_base``, ``result_step``, ``top``,
            ``bottom``, ``hspace``, ``legend_y``.
        """
        s = self.style
        grid_top = s.zoom_grid_top if self.plot_options.zoom else s.grid_top
        top = grid_top - s.top_shrink_per_line * max(self.n_text_lines - 2, 0)

        return {
            "zoom_bounds": [0.0, s.zoom_y0, 1.0, s.zoom_height] if self.plot_options.zoom else None,
            "result_base": s.zoom_y0 + s.zoom_height + s.zoom_text_gap if self.plot_options.zoom
                         else s.result_base,
            "result_step": s.result_step,
            "top": top,
            "bottom": s.bottom,
            "hspace": s.hspace,
            "legend_y": s.legend_bbox[1] + (top - grid_top),
        }

    # ---- base corner drawing ------------------------------------------

    def _get_corner_kwargs(self):
        """Build the keyword arguments shared by every ``corner.corner()`` call."""
        o, s = self.plot_options, self.style
        return {
            "bins":            o.corner_bins,
            "plot_datapoints": o.plot_datapoints,
            "plot_density":    o.plot_2d,
            "plot_contours":   True,
            "fill_contours":   o.plot_2d,
            "levels":          o.levels,
            "labels":          [_latex_label(p) for p in self.data.parameters],
            "range":           self.data.ranges,
            "show_titles":     False,
            "label_kwargs":    {"fontsize": s.label_size},
            "hist_kwargs":     {"histtype": "step", "linewidth": s.hist_linewidth},
            "contour_kwargs":  {"linewidths": s.contour_linewidth},
        }

    def _overlay_run(self, fig, samples, weights, color, smooth, base_kwargs):
        """Overlay one posterior on the shared corner figure and return it."""
        hist_kwargs    = {**base_kwargs["hist_kwargs"],    "color":  color}
        contour_kwargs = {**base_kwargs["contour_kwargs"], "colors": [color]}
        kwargs = {**base_kwargs, "hist_kwargs": hist_kwargs, "contour_kwargs": contour_kwargs}
        if smooth is not None:
            kwargs["smooth"] = smooth
        return corner.corner(samples, fig=fig, weights=weights, color=color, **kwargs)

    def _draw_base_corner(self):
        """Draw every run's posterior on a single shared corner figure.

        In ``concatenated_full`` mode, an overall gray posterior
        (concatenation of all runs) is drawn first; then the highlighted runs
        are redrawn on top.

        Returns
        -------
        (Figure, np.ndarray)
            The figure and a ``(ndim, ndim)`` array of axes.
        """
        base_kwargs = self._get_corner_kwargs()
        smooth = self.plot_options.smooth_2d
        fig = plt.figure(figsize=self.style.figsize)

        if self._is_full_mode:
            all_samples, all_weights = self._all_runs
            fig = self._overlay_run(
                fig, all_samples, all_weights,
                self.style.combined_color,
                None if smooth is None else float(np.nanmean(smooth)),
                base_kwargs,
            )
            order = self.highlighted_order
        else:
            order = self.data.plot_order

        for i in order:
            fig = self._overlay_run(
                fig,
                self.data.samples_by_run[i],
                self.data.weights_by_run[i],
                self.run_color(i),
                None if smooth is None else smooth[i],
                base_kwargs,
            )

        if fig is None:
            raise RuntimeError("corner.corner() did not return a figure.")
        return fig, np.array(fig.axes).reshape((self.data.ndim, self.data.ndim))

    # ---- diagonal histogram modes ------------------------------------

    def _normalize_diagonal_histograms(self):
        """Redraw diagonal panels so every histogram peaks at 1."""
        bins_per_param = [
            np.linspace(*self.data.ranges[j], self.plot_options.hist_bins)
            for j in range(self.data.ndim)
        ]
        for j in range(self.data.ndim):
            ax = self.axes[j, j]
            ax.clear()
            order = self.highlighted_order if self._is_full_mode else self.data.plot_order
            for i in order:
                _draw_step_histogram(
                    ax,
                    self.data.samples_by_run[i][:, j],
                    self.data.weights_by_run[i],
                    bins_per_param[j],
                    self.run_color(i),
                    self.style.hist_linewidth,
                    normalized=True,
                )
            ax.set_xlim(bins_per_param[j][0], bins_per_param[j][-1])
            ax.set_ylim(0, 1.1)
            ax.set_yticks([])

    def _draw_concatenated_histograms(self):
        """Overlay the all-runs concatenated 1D histogram on each diagonal panel."""
        all_samples, all_weights = self._all_runs
        for j in range(self.data.ndim):
            ax = self.axes[j, j]
            bins = np.linspace(*self.data.ranges[j], self.plot_options.hist_bins)
            ymax = _draw_step_histogram(
                ax, all_samples[:, j], all_weights, bins,
                self.style.combined_color,
                self.style.concatenated_hist_linewidth,
                normalized=self.plot_options.normalized,
                zorder=20,
            )
            if ymax > ax.get_ylim()[1]:
                ax.set_ylim(0, 1.1 * ymax)

    def _draw_zoom_histograms(self):
        """Draw zoomed inset 1D histograms above each diagonal panel.

        The first entry of ``plot_order`` (widest posterior) is excluded from
        the zoom view to avoid dominating the inset range.
        """
        style = self.style
        zoom_order = self.data.plot_order[1:]

        for j, reference_value in enumerate(self.data.reference_values):
            ax = self.axes[j, j]
            x = np.concatenate([self.data.samples_by_run[i][:, j] for i in zoom_order])
            weights = np.concatenate([self.data.weights_by_run[i] for i in zoom_order])
            xmin, xmax = _robust_range(
                x, weights, reference_value, style.range_percentiles, style.margin_fraction)
            bins = np.linspace(xmin, xmax, self.plot_options.hist_bins)

            zoom_ax = ax.inset_axes(self.layout["zoom_bounds"])
            ymax = max(
                _draw_step_histogram(
                    zoom_ax,
                    self.data.samples_by_run[i][:, j],
                    self.data.weights_by_run[i],
                    bins, self.run_color(i), style.hist_linewidth,
                    normalized=self.plot_options.normalized,
                )
                for i in zoom_order
            )

            if reference_value is not None:
                zoom_ax.axvline(reference_value, color=style.actual_values_color,
                                linestyle="--", linewidth=style.zoom_reference_linewidth)

            zoom_ax.set_xlim(bins[0], bins[-1])
            zoom_ax.set_ylim(0, 1.05 * ymax if ymax > 0 else 1)
            zoom_ax.set_yticks([])
            zoom_ax.tick_params(axis="both", which="both",
                                bottom=False, left=False,
                                labelbottom=False, labelleft=False)
            for spine in zoom_ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")

            for x_edge in (bins[0], bins[-1]):
                self.fig.add_artist(ConnectionPatch(
                    xyA=(x_edge, 0), xyB=(x_edge, 1),
                    coordsA=zoom_ax.get_xaxis_transform(),
                    coordsB=ax.get_xaxis_transform(),
                    color="black", linewidth=style.zoom_connection_linewidth,
                    clip_on=False, zorder=0,
                ))

    # ---- result texts, markers, legend, axes -------------------------

    def _collect_text_lines(self, parameter_index):
        """Build the list of result-text lines for one diagonal panel.

        Returns
        -------
        list[tuple[float, str, str]]
            Each entry is ``(y_position, color, value_text)`` in transAxes
            units. Lines are ordered top-down (highest y first); the
            combined summary is always at the bottom (``result_base``).
        """
        combined_only = self.mode == "concatenated"
        result_base = self.layout["result_base"]
        result_step = self.layout["result_step"]
        lines = []
        max_decimals = 0

        if not combined_only:
            n_visible = len(self.visible_run_indices)
            for k, i in enumerate(self.visible_run_indices):
                summary = _posterior_summary(
                    self.data.samples_by_run[i][:, parameter_index],
                    self.data.weights_by_run[i],
                )
                value_txt, decimals = _format_summary(*summary)
                max_decimals = max(max_decimals, decimals)
                offset = n_visible - 1 - k + (1 if self._is_full_mode else 0)
                lines.append((result_base + result_step * offset, self.run_color(i), value_txt))

        if combined_only or self._is_full_mode:
            all_samples, all_weights = self._all_runs
            value_txt, decimals = _format_summary(
                *_posterior_summary(all_samples[:, parameter_index], all_weights))
            max_decimals = max(max_decimals, decimals)
            lines.append((result_base, self.style.combined_color, value_txt))

        return lines, max_decimals

    def _add_result_texts(self):
        """Add q50 +- 1sigma summary lines above every diagonal panel.

        Individual-run summaries are stacked top-down in visible order. In
        concatenated mode only the combined summary is shown. In
        concatenated_full mode both per-run and combined are shown.
        """
        for j, parameter in enumerate(self.data.parameters):
            ax = self.axes[j, j]
            label = _latex_label(parameter)
            lines, max_decimals = self._collect_text_lines(j)

            for y, color, value_txt in lines:
                ax.text(0.05, y, rf"{label} $= {value_txt}$",
                        transform=ax.transAxes,
                        color=color, fontsize=self.style.result_fontsize,
                        ha="left", va="baseline", clip_on=False)

            self._add_reference_text(ax, parameter,
                                     self.data.reference_values[j], max_decimals)

    def _add_reference_text(self, ax, parameter, reference_value, decimals):
        """Add the bold reference-value text above a diagonal panel."""
        if reference_value is None:
            return
        label = _latex_label(parameter, bold=True)
        y = self.layout["result_base"] + self.layout["result_step"] * self.n_text_lines
        ax.text(0.05, y,
                rf"{label} $\mathbf{{= {reference_value:.{decimals}f}}}$",
                transform=ax.transAxes,
                color=self.style.actual_values_color,
                fontsize=self.style.result_fontsize,
                ha="left", va="baseline", clip_on=False)

    def _add_reference_markers(self):
        """Add reference lines in 1D panels and reference scatter in 2D panels."""
        for j, reference_value in enumerate(self.data.reference_values):
            if reference_value is not None:
                self.axes[j, j].axvline(
                    reference_value,
                    color=self.style.actual_values_color,
                    linestyle="--", zorder=10,
                    linewidth=self.style.reference_linewidth,
                )

        for row in range(1, self.data.ndim):
            for col in range(row):
                x_ref = self.data.reference_values[col]
                y_ref = self.data.reference_values[row]
                if x_ref is not None and y_ref is not None:
                    self.axes[row, col].scatter(
                        x_ref, y_ref, zorder=10,
                        color=self.style.actual_values_color,
                        s=self.style.reference_marker_size,
                        edgecolor=self.style.reference_marker_edgecolor,
                    )

    def _add_legend(self):
        """Add the figure legend (per-run entries, optional combined, reference)."""
        style = self.style
        handles = [
            Line2D([0], [0], color=self.run_color(i), lw=style.legend_linewidth,
                   label=self.data.labels_by_run[i])
            for i in self.visible_run_indices
        ]

        if self._show_combined:
            label = (f"All runs combined ({self.data.n_runs} exposures)"
                     if self._is_full_mode else "Concatenated")
            handles.append(Line2D([0], [0], color=style.combined_color,
                                  lw=style.legend_linewidth, label=label))

        if self.values is not None:
            handles.append(Line2D([0], [0], color=style.actual_values_color,
                                  linestyle="--", marker="o",
                                  lw=style.reference_linewidth, label="Actual values"))

        self.fig.legend(handles=handles, loc=style.legend_loc,
                        bbox_to_anchor=(style.legend_bbox[0], self.layout["legend_y"]),
                        fontsize=style.legend_fontsize)

    def _format_axes(self):
        """Format tick labels, axis labels, and subplot spacing."""
        last = self.data.ndim - 1

        for (row, _), ax in np.ndenumerate(self.axes):
            ax.tick_params(
                axis="both", which="both",
                left=False, labelleft=False,
                bottom=(row == last), labelbottom=(row == last),
                labelsize=self.style.tick_label_size,
            )

        for col, parameter in enumerate(self.data.parameters):
            ax = self.axes[last, col]
            ax.set_xlabel("")
            for tick_label in ax.get_xticklabels():
                tick_label.set_rotation(45)
                tick_label.set_ha("right")
                tick_label.set_rotation_mode("anchor")
            ax.text(0.5, -0.3 - self.style.x_labelpad,
                    _latex_label(parameter),
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=self.style.label_size, clip_on=False)

        for row in range(1, self.data.ndim):
            ax = self.axes[row, 0]
            ax.set_ylabel(_latex_label(self.data.parameters[row]),
                          fontsize=self.style.label_size)
            ax.yaxis.set_label_coords(self.style.y_labelpad, 0.5)

        self.fig.subplots_adjust(
            top=self.layout["top"],
            bottom=self.layout["bottom"],
            hspace=self.layout["hspace"],
        )


# ============================================================
# PURE HELPERS — statistics and formatting
# ============================================================

def _weighted_percentile(x, weights, percentiles):
    """Return weighted percentiles of ``x``.

    Parameters
    ----------
    x : array-like
    weights : array-like
    percentiles : array-like
        Values in [0, 100].

    Returns
    -------
    np.ndarray
    """
    order = np.argsort(x)
    cdf = np.cumsum(weights[order]) / np.sum(weights)
    return np.interp(np.asarray(percentiles) / 100, cdf, x[order])


def _posterior_summary(samples, weights):
    """Return ``(q50, err_plus, err_minus)`` from weighted percentiles."""
    q16, q50, q84 = _weighted_percentile(samples, weights, [16, 50, 84])
    return q50, q84 - q50, q50 - q16


def _format_summary(q50, err_plus, err_minus):
    """Format a posterior summary as a LaTeX string.

    The smaller of the two errors sets the number of decimal places shown.

    Parameters
    ----------
    q50, err_plus, err_minus : float

    Returns
    -------
    (str, int)
        LaTeX-formatted string and the number of decimal places used.
    """
    err_ref = min(err_plus, err_minus)
    if not np.isfinite(err_ref) or err_ref <= 0:
        decimals = 2
    else:
        decimals = max(0, int(np.ceil(-np.log10(abs(err_ref)))))

    fmt = f".{decimals}f"
    return (rf"{q50:{fmt}}^{{+{err_plus:{fmt}}}}_{{-{err_minus:{fmt}}}}",
            decimals)


def _robust_range(x, weights, reference_value, percentiles, margin_fraction):
    """Weighted-percentile range, expanded to include the reference value.

    A relative margin is added on both sides.

    Parameters
    ----------
    x : array-like
    weights : array-like
    reference_value : float or None
    percentiles : tuple[float, float]
    margin_fraction : float

    Returns
    -------
    (float, float)
        ``(xmin, xmax)``
    """
    xmin, xmax = _weighted_percentile(x, weights, percentiles)
    if reference_value is not None:
        xmin = min(xmin, reference_value)
        xmax = max(xmax, reference_value)
    margin = margin_fraction * ((xmax - xmin) if xmax > xmin else (abs(xmin) or 1.0))
    return xmin - margin, xmax + margin


# ============================================================
# PURE HELPERS — plotting
# ============================================================

def _draw_step_histogram(ax, samples, weights, bins, color, linewidth,
                         normalized=False, **kwargs):
    """Draw a weighted step histogram on ``ax`` and return its maximum value."""
    hist, edges = np.histogram(samples, bins=bins, weights=weights)
    if normalized and hist.max() > 0:
        hist = hist / hist.max()
    ax.step(edges[:-1], hist, where="post",
            color=color, linewidth=linewidth, **kwargs)
    return float(hist.max()) if len(hist) else 0.0


def _latex_label(parameter, bold=False):
    """Return a LaTeX label for a ForMoSA parameter name."""
    if bold:
        label = BOLD_LATEX_LABELS.get(parameter, rf"\mathbf{{{parameter}}}")
    else:
        label = LATEX_LABELS.get(parameter, parameter)
    return rf"${label}$"


# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

def _validate_inputs(config_paths, labels, mode, folder_mode, zoom):
    """Raise informative errors when user inputs are inconsistent."""
    valid = {"concatenated_full", "concatenated", "standard"}
    if mode not in valid:
        raise ValueError(
            f"mode must be one of {sorted(valid)}, got {mode!r}."
        )

    if not config_paths:
        raise ValueError("config_paths must contain at least one path.")

    if folder_mode:
        missing = [str(p) for p in config_paths if not Path(p).is_dir()]
        if missing:
            raise FileNotFoundError(
                "Folder mode is active but some paths are not directories:\n"
                + "\n".join(f"  - {p}" for p in missing)
            )
        empty = [str(p) for p in config_paths if not _find_ini_files_in_folder(p)]
        if empty:
            raise FileNotFoundError(
                "Every folder must contain at least one .ini file. "
                "Empty folders:\n" + "\n".join(f"  - {p}" for p in empty)
            )
    else:
        missing = [str(p) for p in config_paths if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                "The following config paths do not exist or are not files:\n"
                + "\n".join(f"  - {p}" for p in missing)
            )

    if labels is not None and len(labels) != len(config_paths):
        raise ValueError("labels must have the same length as config_paths.")

    if zoom and len(config_paths) < 2:
        raise ValueError("zoom=True requires at least two runs.")


def _load_analysis(config_path):
    """Load one fitted ForMoSA analysis from a ``.ini`` config file."""
    config_path = str(Path(config_path).expanduser())
    config = ConfigLoader(path=config_path, log_level="error").load()
    return Analysis(config["config_path"], adapted=True, fitted=True, log_level="error")


def _find_ini_files_in_folder(config_folder):
    """Return ``.ini`` files directly inside ``config_folder``, sorted by name."""
    folder = Path(config_folder).expanduser()
    return sorted(
        str(path) for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".ini"
    )


def _common_parameters_from_analyses(analyses):
    """Return parameters present in every analysis, in the order of the first.

    Returns
    -------
    (list[str], list[list[str]])
        Common parameter list and the per-run parameter lists.

    Raises
    ------
    ValueError
        If no parameter is common to all runs.
    """
    parameters_by_run = [list(a.ns.results.free_parameters) for a in analyses]
    parameters = [
        p for p in parameters_by_run[0]
        if all(p in run_params for run_params in parameters_by_run)
    ]
    if not parameters:
        raise ValueError("No common free parameters found across config_paths.")
    return parameters, parameters_by_run


def _extract_samples_and_weights(analyses, parameters, parameters_by_run):
    """Extract post-burn-in samples and weights from a list of analyses.

    Parameters
    ----------
    analyses : list[Analysis]
    parameters : list[str]
        Target parameter order (columns of the returned arrays).
    parameters_by_run : list[list[str]]
        Per-run parameter lists (used to locate column indices).

    Returns
    -------
    (list[np.ndarray], list[np.ndarray])
        Samples of shape ``(n_samples, n_parameters)`` and weights of shape
        ``(n_samples,)`` per run.
    """
    samples_by_run, weights_by_run = [], []

    for run_index, (analysis, run_parameters) in enumerate(zip(analyses, parameters_by_run)):
        results = analysis.ns.results
        idx = [run_parameters.index(p) for p in parameters]
        samples = np.asarray(results.samples[results.burn_in:, idx], dtype=float)
        weights = np.asarray(results.weights[results.burn_in:], dtype=float)

        if samples.shape[0] != weights.shape[0]:
            raise ValueError(f"Run {run_index + 1}: samples and weights have different lengths.")
        if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(weights)):
            raise ValueError(f"Run {run_index + 1}: samples or weights contain NaN/inf.")
        if np.any(weights < 0) or np.sum(weights) <= 0:
            raise ValueError(f"Run {run_index + 1}: invalid weights.")

        samples_by_run.append(samples)
        weights_by_run.append(weights)

    return samples_by_run, weights_by_run


def _select_parameters(parameters, samples_by_run, selected_params):
    """Restrict parameters (and matching sample columns) to ``selected_params``.

    Parameters
    ----------
    parameters : list[str]
    samples_by_run : list[np.ndarray]
    selected_params : list[int] or None
        Indices to keep, e.g. ``[1, 2]``. ``None`` keeps everything.

    Returns
    -------
    (list[str], list[np.ndarray])
    """
    if selected_params is None:
        return parameters, samples_by_run

    if len(selected_params) < 2:
        raise ValueError("selected_params must contain at least two indices.")

    out_of_range = [i for i in selected_params if i < 0 or i >= len(parameters)]
    if out_of_range:
        raise ValueError(
            f"selected_params indices out of range for {len(parameters)} parameters: "
            f"{out_of_range}."
        )

    return (
        [parameters[i] for i in selected_params],
        [samples[:, selected_params] for samples in samples_by_run],
    )


def _compute_plot_order(samples_by_run, weights_by_run, percentiles):
    """Return run indices sorted widest-to-narrowest posterior (widest drawn first).

    Width is measured as the mean inter-percentile range across all parameters.
    Wider runs are drawn first so narrower posteriors appear on top.

    Parameters
    ----------
    samples_by_run : list[np.ndarray], shape (n_samples, n_parameters) per run
    weights_by_run : list[np.ndarray], shape (n_samples,) per run
    percentiles : tuple[float, float]
        Percentile pair used to measure posterior width, e.g. ``(16, 84)``.

    Returns
    -------
    list[int]
        Run indices from widest to narrowest.
    """
    widths = [
        np.nanmean([
            np.diff(_weighted_percentile(s[:, j], w, percentiles))[0]
            for j in range(s.shape[1])
        ])
        for s, w in zip(samples_by_run, weights_by_run)
    ]
    return list(np.argsort(widths)[::-1])


def _build_corner_data(samples_by_run, weights_by_run, labels_by_run, parameters,
                       values, style, plot_options, cycle):
    """Assemble a ``CornerData`` object from prepared posterior samples."""
    parameters, samples_by_run = _select_parameters(
        parameters, samples_by_run, plot_options.selected_params)

    reference_values = [values.get(p) if values else None for p in parameters]

    all_weights = np.concatenate(weights_by_run)
    ranges = [
        _robust_range(
            np.concatenate([samples[:, j] for samples in samples_by_run]),
            all_weights, reference_values[j],
            style.range_percentiles, style.margin_fraction,
        )
        for j in range(len(parameters))
    ]

    return CornerData(
        parameters=parameters,
        reference_values=reference_values,
        samples_by_run=samples_by_run,
        weights_by_run=weights_by_run,
        labels_by_run=labels_by_run,
        colors_by_run=style.colors_by_run or [cycle[i % len(cycle)]
                                               for i in range(len(samples_by_run))],
        ranges=ranges,
        plot_order=_compute_plot_order(
            samples_by_run, weights_by_run, style.plot_order_percentiles),
    )


def _prepare_corner_data_from_files(config_paths, labels, values, style, plot_options, cycle):
    """Load analyses from individual ``.ini`` files and build ``CornerData``."""
    analyses = [_load_analysis(path) for path in config_paths]
    parameters, parameters_by_run = _common_parameters_from_analyses(analyses)
    samples_by_run, weights_by_run = _extract_samples_and_weights(
        analyses, parameters, parameters_by_run)
    labels_by_run = labels or [f"Run {i + 1}" for i in range(len(analyses))]
    return _build_corner_data(samples_by_run, weights_by_run, labels_by_run,
                              parameters, values, style, plot_options, cycle)


def _prepare_corner_data_from_folders(config_folders, labels, values, style, plot_options, cycle):
    """Load analyses from folders, concatenate runs inside each folder.

    Each folder becomes one plotted run. All ``.ini`` files found directly
    inside a folder are loaded, their posterior samples concatenated, and the
    result treated as a single run.
    """
    grouped_paths = [_find_ini_files_in_folder(f) for f in config_folders]
    analyses_by_group = [[_load_analysis(p) for p in paths] for paths in grouped_paths]
    all_analyses = [a for group in analyses_by_group for a in group]

    parameters, _ = _common_parameters_from_analyses(all_analyses)

    samples_by_run, weights_by_run = [], []
    for analyses in analyses_by_group:
        parameters_by_run = [list(a.ns.results.free_parameters) for a in analyses]
        group_samples, group_weights = _extract_samples_and_weights(
            analyses, parameters, parameters_by_run)
        samples_by_run.append(np.concatenate(group_samples, axis=0))
        weights_by_run.append(np.concatenate(group_weights, axis=0))

    labels_by_run = labels or [
        f"{Path(folder).name} ({len(paths)} configs)"
        for folder, paths in zip(config_folders, grouped_paths)
    ]
    return _build_corner_data(samples_by_run, weights_by_run, labels_by_run,
                              parameters, values, style, plot_options, cycle)


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def plot_corner_from_config_paths(
    config_paths,
    labels=None,
    values=None,
    mode="standard",
    plot_options=None,
    style=None,
) -> Figure:
    """
    Compare several ForMoSA posterior runs on a single corner plot.

    This is the main entry point of the module. It loads the fitted analyses,
    extracts posterior samples, and produces a fully decorated corner figure
    with per-run result texts, optional reference markers, and a legend.

    Parameters
    ----------
    config_paths : list[str]
        Paths to the ForMoSA ``.ini`` config files, or paths to *folders*
        containing ``.ini`` files. Folder mode is inferred automatically: if
        the first entry is a directory, all entries are treated as folders and
        samples within each folder are concatenated into a single run.

    labels : list[str], optional
        Legend label for each run (or folder when folder mode is inferred).
        Must have the same length as ``config_paths``. If omitted, labels
        default to ``"Run 1"``, ``"Run 2"``, etc.

    values : dict, optional
        Reference (true / injected) values for any subset of the free
        parameters, e.g. ``{"Teff": 1800, "log(g)": 4.0, "C/O": 0.55}``.
        Parameters not in the dict receive no reference marker.

    mode : str, default ``"standard"``
        Controls the diagonal histogram overlay and which runs are highlighted.
        Must be one of:

        ``"standard"``
            One step histogram per run, no overlay.
        ``"concatenated"``
            Adds an all-runs concatenated posterior on the diagonal in a
            darker colour; only the combined summary text is shown.
        ``"concatenated_full"``
            Draws a gray combined posterior in the background, then overlays
            the first *n_highlighted_runs* runs in colour on top. Shows both
            per-run and combined summary texts.

    plot_options : CornerPlotOptions, optional
        Fine-grained switches: bin counts, smoothing, highlighted-run count,
        parameter selection, contour levels, zoom insets, etc. See
        ``CornerPlotOptions`` for the full list of attributes. Defaults to
        ``CornerPlotOptions()``.

    style : CornerPlotStyle, optional
        All visual and layout properties: figure size, font sizes, colours,
        linewidths, spacing, legend placement. See ``CornerPlotStyle`` for the
        full list of attributes. Defaults to ``CornerPlotStyle()``.

    Returns
    -------
    matplotlib.figure.Figure
        The completed corner figure. Call ``fig.savefig(...)`` to export.

    Raises
    ------
    ValueError
        If ``mode`` is not recognised, if ``config_paths`` is empty,
        if ``labels`` has the wrong length, if ``zoom=True`` with fewer than
        two runs, or if no common parameters are found.
    FileNotFoundError
        If any path in ``config_paths`` does not exist, or if folder mode is
        inferred but a path is not a directory or contains no ``.ini`` files.

    Examples
    --------
    Basic three-run comparison::

        fig = plot_corner_from_config_paths(
            config_paths=["run1.ini", "run2.ini", "run3.ini"],
            labels=["Low contrast", "Medium contrast", "High contrast"],
            values={"Teff": 1800, "log(g)": 4.0},
            mode="concatenated",
        )
        fig.savefig("corner.pdf", bbox_inches="tight")

    Restrict the plot to two parameters and customise the style::

        fig = plot_corner_from_config_paths(
            config_paths=["run1.ini", "run2.ini"],
            plot_options=CornerPlotOptions(selected_params=[0, 2]),
            style=CornerPlotStyle(figsize=(8, 8), legend_fontsize=12),
        )
    """
    plot_options = plot_options or CornerPlotOptions()
    style = style or CornerPlotStyle()
    folder_mode = Path(config_paths[0]).is_dir() if config_paths else False
    _validate_inputs(config_paths, labels, mode, folder_mode, plot_options.zoom)

    with plt.rc_context({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix"}):
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
        if folder_mode:
            data = _prepare_corner_data_from_folders(
                config_paths, labels, values, style, plot_options, cycle)
        else:
            data = _prepare_corner_data_from_files(
                config_paths, labels, values, style, plot_options, cycle)

        fig = CornerPlotter(
            data=data,
            values=values,
            mode=mode,
            plot_options=plot_options,
            style=style,
        ).plot()

        return fig