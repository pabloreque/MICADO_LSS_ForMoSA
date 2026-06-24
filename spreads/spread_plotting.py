#!/usr/bin/env python3
"""
Plot saved ForMoSA spread and parameter-variation ``.npz`` files.

Public entry points: ``plot_spread()``, ``plot_multiple_spreads()``,
``build_variation_cache()``, ``plot_variation_by_parameter()``,
``plot_variation_stacked()``.

Overview
--------
All functions are designed to be imported and called from Jupyter notebooks.
They read compact ``.npz`` caches produced by ``spread_builders.py`` and render
Matplotlib figures without side effects on import.

Two families of plots are provided:

*Spread plots* — posterior or noise envelopes
    ``plot_spread`` and ``plot_multiple_spreads`` read files produced by
    ``physical_posterior_spread`` or ``covariance_noise_spread`` and draw
    1/2/3-sigma shaded bands together with the best-fit model and observed data.

*Parameter-variation plots* — sensitivity to individual parameters
    ``build_variation_cache`` first builds a ``.npz`` cache by evaluating the
    forward model at the initial and the fitted value for each free parameter.
    ``plot_variation_by_parameter`` and ``plot_variation_stacked`` then render
    per-parameter sensitivity figures from that cache.

Typical notebook usage
----------------------
Spread plot::

    from spread_plotting import plot_spread

    fig, ax = plot_spread("posterior_spread.npz")
    display(fig)

Comparison of multiple runs::

    from spread_plotting import plot_multiple_spreads, SpreadPlotStyle

    fig, axes = plot_multiple_spreads(
        ["run1.npz", "run2.npz"],
        labels=["Low contrast", "High contrast"],
        style=SpreadPlotStyle(model_color=["tab:blue", "tab:orange"]),
    )
    display(fig)

Parameter-variation plot::

    from spread_plotting import build_variation_cache, plot_variation_stacked

    cache = build_variation_cache(
        config_path="config.ini",
        output_path="variation_cache.npz",
        initial_model_values={"Teff": 1800, "log(g)": 4.0},
    )
    fig, ax, ax_right = plot_variation_stacked(cache)
    display(fig)
"""

from dataclasses import dataclass, fields, replace
from pathlib import Path
import colorsys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


# ============================================================
# CONFIGURATION DATACLASSES
# ============================================================


@dataclass
class SpreadPlotOptions:
    """
    Plot-level switches shared by all cases drawn in the same figure.

    These options decide which plot components are visible and how multiple
    cases are arranged.

    Attributes
    ----------
    plot_median : bool
        If True, draw the posterior median flux curve.
    plot_best_fit : bool
        If True, draw the best-fit model curve.
    show_data : bool
        If True, draw the observed data stored in the ``.npz`` cache.
    show_sigma_labels : bool
        If True, add ``"1σ"``, ``"2σ"``, ``"3σ"`` labels to the shaded
        envelope regions in the legend.
    layout : str
        ``"overlay"`` draws all cases on a single axis.
        ``"side_by_side"`` creates one axis per case.
        Only used by ``plot_multiple_spreads``.
    """

    plot_median: bool = True
    plot_best_fit: bool = True
    show_data: bool = True
    show_sigma_labels: bool = False
    layout: str = "overlay"


@dataclass
class SpreadPlotStyle:
    """
    Style options for one or more spread cases.

    Most fields accept either a single value (applied to every case) or a
    list/tuple with one value per case.  For example,
    ``model_color="tab:blue"`` uses the same colour for all cases, while
    ``model_color=["tab:blue", "tab:orange"]`` assigns one colour to each
    case in ``plot_multiple_spreads``.

    Attributes
    ----------
    label_prefix : str or list[str]
        Text prepended to the median and best-fit legend labels.  Usually the
        case name when comparing runs in overlay mode.
    model_color : str, list[str], or None
        Colour used for the model line and all sigma envelopes.  ``None``
        picks colours automatically from the active Matplotlib colour cycle.
    data_color : str or list[str]
        Colour used for the observed-data curve.
    median_linewidth : float or list[float]
        Line width of the posterior median curve.
    best_fit_linewidth : float or list[float]
        Line width of the best-fit model curve.
    data_linewidth : float or list[float]
        Line width of the observed-data curve.
    median_linestyle : str or list[str]
        Line style of the posterior median curve.
    best_fit_linestyle : str or list[str]
        Line style of the best-fit model curve.
    data_linestyle : str or list[str]
        Line style of the observed-data curve.
    data_marker : str, list[str or None], or None
        Marker style for the observed-data curve.
    data_markersize : float, list[float or None], or None
        Marker size for the observed-data curve.
    data_alpha : float or list[float]
        Opacity of the observed-data curve.
    sigma_1_alpha : float or list[float]
        Opacity of the 1-sigma shaded envelope.
    sigma_2_alpha : float or list[float]
        Opacity of the 2-sigma shaded envelope.
    sigma_3_alpha : float or list[float]
        Opacity of the 3-sigma shaded envelope.
    """

    label_prefix: str | list[str] | tuple[str, ...] = ""

    model_color: str | list[str] | tuple[str, ...] | None = None
    data_color: str | list[str] | tuple[str, ...] = "0.55"

    median_linewidth: float | list[float] | tuple[float, ...] = 1.1
    best_fit_linewidth: float | list[float] | tuple[float, ...] = 0.9
    data_linewidth: float | list[float] | tuple[float, ...] = 0.9

    median_linestyle: str | list[str] | tuple[str, ...] = "--"
    best_fit_linestyle: str | list[str] | tuple[str, ...] = "-"
    data_linestyle: str | list[str] | tuple[str, ...] = "-"
    data_marker: str | list[str | None] | tuple[str | None, ...] | None = None
    data_markersize: float | list[float | None] | tuple[float | None, ...] | None = 3

    data_alpha: float | list[float] | tuple[float, ...] = 0.9
    sigma_1_alpha: float | list[float] | tuple[float, ...] = 0.35
    sigma_2_alpha: float | list[float] | tuple[float, ...] = 0.25
    sigma_3_alpha: float | list[float] | tuple[float, ...] = 0.15


@dataclass
class AxisOptions:
    """
    Axis-formatting options shared by all spread plotting helpers.

    Attributes
    ----------
    x_lim : tuple[float, float] or None
        Wavelength range ``(min, max)`` in µm.  ``None`` uses the full data
        range.
    y_lim : tuple[float, float], list[tuple], or None
        Flux axis limits.  When ``plot_multiple_spreads`` is used with
        ``layout="side_by_side"``, a list of one tuple per case applies
        independent y limits to each panel.
    figsize : tuple[float, float] or None
        Overall figure size in inches ``(width, height)``.  Used by
        ``plot_spread`` and the overlay mode of ``plot_multiple_spreads``.
    figsize_per_panel : tuple[float, float]
        Figure size per panel used by ``plot_multiple_spreads`` with
        ``layout="side_by_side"``.  The final figure width is
        ``figsize_per_panel[0] * n_cases``.
    xlabel : str or None
        X-axis label.  If ``None``, the plotting helpers generate a
        wavelength label and append the bin count when binning is active.
    ylabel : str or None
        Y-axis label.  Set to ``None`` to suppress.
    label_size : float
        Font size for axis labels.
    title_size : float
        Font size for panel titles in side-by-side mode.
    tick_label_size : float
        Font size for tick labels.
    show_legend : bool
        If False, the legend is suppressed entirely.
    legend_loc : str
        ``loc`` argument passed to ``ax.legend()``.
    legend_fontsize : float
        Font size for legend entries.
    legend_alpha : float
        Frame opacity of the legend box.  Set to 0 to hide the frame.
    legend_linewidth : float or None
        If set, overrides the line width of every ``Line2D`` handle in the
        legend so that thin plot lines remain visible.  ``None`` keeps the
        original handle width.
    """

    x_lim: tuple[float, float] | None = None
    y_lim: tuple[float, float] | list[tuple[float, float]] | None = None
    figsize: tuple[float, float] | None = (15, 5)
    figsize_per_panel: tuple[float, float] = (5, 5)

    xlabel: str | None = None
    ylabel: str | None = "Flux (W/m²/µm)"

    label_size: float = 18
    title_size: float = 18
    tick_label_size: float = 14

    show_legend: bool = True
    legend_loc: str = "upper right"
    legend_fontsize: float = 14
    legend_alpha: float = 0.9
    legend_linewidth: float | None = 2.5


# Keys for flux arrays present in every saved item.
MODEL_FLUX_KEYS = [
    "best_fit",
    "median",
    "spread_1_low",
    "spread_1_high",
    "spread_2_low",
    "spread_2_high",
    "spread_3_low",
    "spread_3_high",
]

FLUX_KEYS = MODEL_FLUX_KEYS + ["obs_flux", "obs_flux_all"]

# Matplotlib style applied inside rc_context for all public plot functions.
_MPL_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 18,
}



# ============================================================
# BASIC HELPERS
# ============================================================


def _bin_mean(x, y, bins):
    """Return finite mean-binned ``x`` and ``y`` arrays.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (wavelength).
    y : np.ndarray
        Dependent variable (flux).
    bins : None, int, or array-like
        Binning specification:

        - ``None`` — no binning; only non-finite points are removed.
        - int — build that many equally spaced bins over the finite ``x`` range.
        - array — use those values directly as bin edges.

    Returns
    -------
    x_binned : np.ndarray
        Mean ``x`` value in each non-empty bin (or the filtered ``x`` if
        ``bins`` is ``None``).
    y_binned : np.ndarray
        Mean ``y`` value in each non-empty bin.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if bins is None:
        return x, y

    if x.size == 0:
        return np.array([]), np.array([])

    bins = (np.linspace(np.nanmin(x), np.nanmax(x), int(bins) + 1)
            if np.isscalar(bins) else np.asarray(bins, dtype=float))

    bin_index = np.digitize(x, bins) - 1
    x_bin, y_bin = [], []

    for i in range(len(bins) - 1):
        in_bin = bin_index == i  # Mask selecting the points that fall into this bin.
        if np.any(in_bin):
            x_bin.append(np.nanmean(x[in_bin]))
            y_bin.append(np.nanmean(y[in_bin]))

    return np.asarray(x_bin), np.asarray(y_bin)


def _saturate_colors(colors):
    """Return a copy of ``colors`` with HSL saturation boosted by 35%.

    Used to keep model envelopes visually distinct when they are drawn with
    transparency on top of each other.

    Parameters
    ----------
    colors : list
        Matplotlib-compatible colour specifications.

    Returns
    -------
    list
        New colour list with the same hue and lightness but saturation capped
        at ``min(1.0, 1.35 * s)`` for each entry.
    """
    result = []
    for color in colors:
        r, g, b = to_rgba(color)[:3]
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        result.append(colorsys.hls_to_rgb(h, l, min(1.0, 1.35 * s)))
    return result


def _default_model_colors(n_cases):
    """Return one saturated Matplotlib colour per case.

    Colours are taken from the active ``axes.prop_cycle`` and repeated modulo
    the cycle length when ``n_cases`` exceeds the number of available colours.
    Saturation is boosted so that transparent sigma envelopes retain visible
    contrast.

    Parameters
    ----------
    n_cases : int
        Number of colours to generate.

    Returns
    -------
    list[str]
        List of ``n_cases`` colour strings.
    """
    saturated = _saturate_colors(
        plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
    )
    return [saturated[i % len(saturated)] for i in range(n_cases)]


def _resolve_case_style_values(style, n_cases):
    """Expand every ``SpreadPlotStyle`` field to a list of length ``n_cases``.

    Public plotting functions accept both scalar style values and per-case
    lists.  This helper normalises everything to plain lists so the rest of the
    plotting code can index by case without branching.

    ``model_color=None`` is resolved here by calling
    :func:`_default_model_colors`.

    Parameters
    ----------
    style : SpreadPlotStyle
        Style object whose fields may be scalars or lists.
    n_cases : int
        Number of cases being plotted.

    Returns
    -------
    dict[str, list]
        Mapping from field name to a list of ``n_cases`` values.

    Raises
    ------
    ValueError
        If a list-valued field has a length different from ``n_cases``.

    Examples
    --------
    ``SpreadPlotStyle(label_prefix="run")`` becomes::

        {"label_prefix": ["run", "run", ...], ...}

    while ``SpreadPlotStyle(label_prefix=["run 1", "run 2"])`` is kept as::

        {"label_prefix": ["run 1", "run 2"], ...}
    """
    case_style_values = {}

    for field in fields(SpreadPlotStyle):
        name = field.name
        value = getattr(style, name)

        if name == "model_color" and value is None:
            case_style_values[name] = _default_model_colors(n_cases)
        elif isinstance(value, (list, tuple)):
            if len(value) != n_cases:
                raise ValueError(
                    f"Expected {n_cases} {name.replace('_', ' ')}, got {len(value)}.")
            case_style_values[name] = list(value)
        else:
            case_style_values[name] = [value] * n_cases

    return case_style_values


def _format_spread_axis(ax, bins, axis_options):
    """Apply axis limits, labels, ticks, and a de-duplicated legend.

    Intended to be called once after all items have been drawn on ``ax``.
    Legend entries are de-duplicated by label so repeated observations do not
    produce many identical ``"data"`` or ``"1σ"`` entries; the first occurrence
    of each label wins.

    When ``axis_options.legend_linewidth`` is set, the line width of every
    ``Line2D`` legend handle is overridden to that value so that thin spectral
    lines remain identifiable in the legend box.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to format.
    bins : None, int, or array-like
        Binning specification forwarded from the calling plot function.  Used
        to construct the default x-axis label when ``axis_options.xlabel`` is
        ``None``.
    axis_options : AxisOptions
        Axis formatting options.
    """
    if axis_options.x_lim is not None:
        ax.set_xlim(*axis_options.x_lim)

    if axis_options.y_lim is not None:
        ax.set_ylim(*axis_options.y_lim)

    xlabel = axis_options.xlabel
    if xlabel is None:
        n_bins = int(bins) if np.isscalar(bins) else len(bins) - 1 if bins is not None else None
        xlabel = ("Wavelength (µm)" if bins is None
                  else f"Wavelength (µm) ({n_bins} bins)")

    ax.set_xlabel(xlabel, fontsize=axis_options.label_size)

    if axis_options.ylabel is not None:
        ax.set_ylabel(axis_options.ylabel, fontsize=axis_options.label_size)

    ax.tick_params(direction="in", top=True, right=True,
                   labelsize=axis_options.tick_label_size)

    if not axis_options.show_legend:
        return

    # De-duplicate legend labels (e.g. "data" repeated per observation).
    # Iterate forward so the first occurrence of each label wins.
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_") and label not in unique:
            unique[label] = handle

    if not unique:
        return

    legend = ax.legend(list(unique.values()), list(unique.keys()),
                       loc=axis_options.legend_loc, fontsize=axis_options.legend_fontsize,
                       frameon=axis_options.legend_alpha > 0,
                       framealpha=axis_options.legend_alpha)

    # Thicken legend line handles without affecting the actual plot lines.
    if axis_options.legend_linewidth is not None:
        for handle in legend.legend_handles:
            if hasattr(handle, "set_linewidth"):
                handle.set_linewidth(axis_options.legend_linewidth)


# ============================================================
# LOADING
# ============================================================


def _load_spread_items_from_npz(npz_path, case_label=None):
    """Load all observation dicts from one spread ``.npz`` file.

    Each returned dict is a shallow copy of the corresponding ``all_mods``
    entry enriched with two metadata keys: ``npz_path`` (source path as a
    string) and ``case_label`` (label used when comparing several caches).

    Parameters
    ----------
    npz_path : str or Path
        Path to a ``.npz`` file produced by ``spread_builders``.
    case_label : str or None
        Label to attach to each item.  Defaults to the string form of
        ``npz_path`` when ``None``.

    Returns
    -------
    list[dict]
        One dict per observation stored in the file.
    """
    data = np.load(npz_path, allow_pickle=True)
    label = case_label if case_label is not None else str(npz_path)

    return [
        {**dict(model_dict), "npz_path": str(npz_path), "case_label": label}
        for model_dict in data["all_mods"]
    ]


# ============================================================
# DATA PREPARATION
# ============================================================


def _load_model_curves(item, x_lim=None, bins=None):
    """Extract model curves from one saved observation item.

    The model wavelength grid is restricted to ``x_lim`` and then optionally
    binned via :func:`_bin_mean`.  Every key in ``MODEL_FLUX_KEYS`` is
    returned; missing or ``None`` values in the item are stored as ``None`` so
    callers can skip unavailable components without branching.

    Parameters
    ----------
    item : dict
        One entry from ``all_mods`` as loaded by
        :func:`_load_spread_items_from_npz`.
    x_lim : tuple[float, float] or None
        Wavelength range to restrict to.  ``None`` keeps all points.
    bins : None, int, or array-like
        Binning passed to :func:`_bin_mean`.

    Returns
    -------
    tuple[np.ndarray, dict] or None
        ``(x_model, curves)`` where ``x_model`` is the (possibly binned)
        wavelength grid and ``curves`` maps each ``MODEL_FLUX_KEYS`` key to a
        flux array or ``None``.  Returns ``None`` when no wavelength points
        survive the ``x_lim`` cut.
    """
    wav = np.asarray(item["wav"], dtype=float)
    mask = np.ones_like(wav, dtype=bool) if x_lim is None else (wav >= x_lim[0]) & (wav <= x_lim[1])
    wav_cut = wav[mask]

    if wav_cut.size == 0:
        return None

    # Use _bin_mean on the wavelength array itself to get the binned x grid.
    x_model, _ = _bin_mean(wav_cut, wav_cut, bins)

    curves = {}
    for key in MODEL_FLUX_KEYS:
        if key not in item or item[key] is None:
            curves[key] = None
            continue
        x_model, curves[key] = _bin_mean(wav_cut, np.asarray(item[key], dtype=float)[mask], bins)

    return x_model, curves


def _load_data_curves(item, x_lim=None, bins=None):
    """Extract observed-data curves from one saved observation item.

    Two cache formats are handled transparently:

    - ``obs_wave_all`` / ``obs_flux_all``: multiple observations stored as 2D
      arrays (shape ``(M, N)``).  Each row is returned as a separate curve.
    - ``obs_wave`` / ``obs_flux``: a single observation curve.

    Parameters
    ----------
    item : dict
        One entry from ``all_mods``.
    x_lim : tuple[float, float] or None
        Wavelength range to restrict to.
    bins : None, int, or array-like
        Binning passed to :func:`_bin_mean`.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        One ``(x, y)`` pair per observation curve that survives the
        ``x_lim`` cut.  Empty when no observational data is stored.

    Raises
    ------
    ValueError
        If ``obs_wave_all`` and ``obs_flux_all`` are not both 2D or have
        mismatched shapes.
    """
    if "obs_wave_all" in item and "obs_flux_all" in item:
        obs_wave_all = np.asarray(item["obs_wave_all"], dtype=float)
        obs_flux_all = np.asarray(item["obs_flux_all"], dtype=float)

        if obs_wave_all.ndim != 2 or obs_flux_all.ndim != 2:
            raise ValueError("obs_wave_all and obs_flux_all must both be 2D arrays.")

        if obs_wave_all.shape != obs_flux_all.shape:
            raise ValueError(
                f"obs_wave_all and obs_flux_all shape mismatch: "
                f"{obs_wave_all.shape} vs {obs_flux_all.shape}"
            )

        result = []
        for obs_wave, obs_flux in zip(obs_wave_all, obs_flux_all, strict=True):
            x, y = np.asarray(obs_wave, dtype=float), np.asarray(obs_flux, dtype=float)
            if x_lim is not None:
                m = (x >= x_lim[0]) & (x <= x_lim[1])
                x, y = x[m], y[m]
            if x.size > 0:
                result.append(_bin_mean(x, y, bins))
        return result

    if "obs_wave" not in item or "obs_flux" not in item:
        return []

    x_data = np.asarray(item["obs_wave"], dtype=float)
    y_data = np.asarray(item["obs_flux"], dtype=float)
    if x_lim is not None:
        m = (x_data >= x_lim[0]) & (x_data <= x_lim[1])
        x_data, y_data = x_data[m], y_data[m]
    x_data, y_data = _bin_mean(x_data, y_data, bins)
    return [(x_data, y_data)] if x_data.size > 0 else []


def _scale_fluxes(item, scale, offset):
    """Return a copy of an item with all flux arrays scaled and offset.

    Applies ``scaled_flux = original_flux * scale + offset`` to every key in
    ``FLUX_KEYS`` that is present and non-``None``.  The original ``item`` is
    not modified.

    Parameters
    ----------
    item : dict
        One entry from ``all_mods``.
    scale : float
        Multiplicative factor applied to every flux array.
    offset : float
        Additive offset applied after scaling.

    Returns
    -------
    dict
        Shallow copy of ``item`` with flux arrays replaced by their scaled
        versions.
    """
    item_scaled = dict(item)
    for key in FLUX_KEYS:
        if key in item_scaled and item_scaled[key] is not None:
            item_scaled[key] = np.asarray(item_scaled[key], dtype=float) * scale + offset
    return item_scaled


# ============================================================
# PLOTTING ONE ITEM ON AN EXISTING AXIS
# ============================================================


def _plot_one_spread_item_from_npz(ax, item, bins, style, plot_options, axis_options,
                                   label_data=False):
    """Plot one saved observation item on an existing Matplotlib axis.

    Low-level routine called by :func:`_plot_spread_case`.  Draws, depending
    on ``plot_options``:

    - the observed data (from ``obs_wave`` / ``obs_flux`` or the ``_all``
      variants);
    - the 3σ, 2σ, and 1σ shaded envelopes;
    - the posterior median model curve;
    - the best-fit model curve.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    item : dict
        One entry from ``all_mods`` (may already be scaled).
    bins : None, int, or array-like
        Wavelength binning.
    style : SpreadPlotStyle
        Style for this specific case (scalar fields, already resolved).
    plot_options : SpreadPlotOptions
        Visibility switches.
    axis_options : AxisOptions
        Used for ``x_lim`` when loading curves.
    label_data : bool
        If True, this call adds the ``"data"`` label to the legend.  Should
        be ``True`` only for the first data curve drawn across all cases.
    """
    loaded = _load_model_curves(item, x_lim=axis_options.x_lim, bins=bins)

    if loaded is None:
        print(f"Skipping {item['observation']}: "
              f"no model points inside x_lim={axis_options.x_lim}")
        return

    x_model, curves = loaded

    if plot_options.show_data:
        for i, (x_data, y_data) in enumerate(
                _load_data_curves(item, x_lim=axis_options.x_lim, bins=bins)):
            ax.plot(x_data, y_data, color=style.data_color, linewidth=style.data_linewidth,
                    linestyle=style.data_linestyle, marker=style.data_marker,
                    markersize=style.data_markersize, alpha=style.data_alpha,
                    label="data" if label_data and i == 0 else None,
                    rasterized=True, zorder=1)

    for sigma, low_key, high_key, alpha in (
        ("3σ", "spread_3_low", "spread_3_high", style.sigma_3_alpha),
        ("2σ", "spread_2_low", "spread_2_high", style.sigma_2_alpha),
        ("1σ", "spread_1_low", "spread_1_high", style.sigma_1_alpha),
    ):
        if curves[low_key] is None or curves[high_key] is None:
            continue

        ax.fill_between(x_model, curves[low_key], curves[high_key], color=style.model_color,
                        alpha=alpha, linewidth=0,
                        label=sigma if plot_options.show_sigma_labels else None, zorder=2)

    if plot_options.plot_median and curves["median"] is not None:
        ax.plot(x_model, curves["median"], color=style.model_color,
                linewidth=style.median_linewidth, linestyle=style.median_linestyle,
                label=f"{style.label_prefix} median".strip(), rasterized=True, zorder=4)

    if plot_options.plot_best_fit and curves["best_fit"] is not None:
        ax.plot(x_model, curves["best_fit"], color=style.model_color,
                linewidth=style.best_fit_linewidth, linestyle=style.best_fit_linestyle,
                label=f"{style.label_prefix} best-fit".strip(), rasterized=True, zorder=5)


def _plot_spread_case(ax, items, bins, style, plot_options, axis_options,
                      data_label_used=False):
    """Plot all observations belonging to one spread case.

    Iterates over ``items`` (one per observation) and calls
    :func:`_plot_one_spread_item_from_npz` for each.  Tracks whether the
    ``"data"`` legend label has been used so that overlay plots with many
    observations across many cases produce only one data entry.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    items : list[dict]
        All observation entries for this case.
    bins : None, int, or array-like
        Wavelength binning.
    style : SpreadPlotStyle
        Style for this case (scalar fields, already resolved).
    plot_options : SpreadPlotOptions
        Visibility switches.
    axis_options : AxisOptions
        Axis formatting options.
    data_label_used : bool
        Whether ``"data"`` has already been added to the legend by a previous
        case.

    Returns
    -------
    bool
        Updated ``data_label_used`` flag after plotting this case.
    """
    for i, item in enumerate(items):
        label_data = plot_options.show_data and not data_label_used
        _plot_one_spread_item_from_npz(
            ax=ax, item=item, bins=bins, style=style,
            plot_options=replace(plot_options,
                                 show_sigma_labels=plot_options.show_sigma_labels and i == 0),
            axis_options=axis_options, label_data=label_data,
        )
        if label_data:
            data_label_used = True

    return data_label_used


# ============================================================
# PLOT ONE .NPZ FILE
# ============================================================


def plot_spread(
    npz_path,
    bins=None,
    style=None,
    axis_options=None,
    plot_options=None,
):
    """Plot every observation stored in one spread ``.npz`` file.

    Parameters
    ----------
    npz_path : str or Path
        Path to a spread cache produced by ``spread_builders``
        (``all_mods`` array of observation dicts).
    bins : None, int, or array-like, default None
        Wavelength binning.  ``None`` for no binning, an integer for that
        many equally spaced bins, or an array of custom bin edges.
    style : SpreadPlotStyle or None, default None
        Colours, line styles, linewidths, markers, and alpha values.
        Defaults to ``SpreadPlotStyle()``.
    axis_options : AxisOptions or None, default None
        Limits, labels, figure size, and legend settings.
        Defaults to ``AxisOptions()``.
    plot_options : SpreadPlotOptions or None, default None
        Visibility switches for median, best-fit, data, sigma labels.
        Defaults to ``SpreadPlotOptions()``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The completed figure and its single axis.

    Examples
    --------
    ::

        fig, ax = plot_spread(
            "physical_posterior_spread.npz",
            bins=50,
            style=SpreadPlotStyle(model_color="tab:blue"),
        )
        display(fig)
    """
    style = style or SpreadPlotStyle()
    plot_options = plot_options or SpreadPlotOptions()
    axis_options = axis_options or AxisOptions()

    case_style_values = _resolve_case_style_values(style, 1)
    style = replace(style, **{f: v[0] for f, v in case_style_values.items()})

    items = _load_spread_items_from_npz(npz_path)
    print(f"Loaded {len(items)} observations from: {npz_path}")

    with plt.rc_context(_MPL_RC):
        fig, ax = plt.subplots(figsize=axis_options.figsize)
        _plot_spread_case(ax, items, bins, style, plot_options, axis_options)
        _format_spread_axis(ax, bins, axis_options)
        fig.tight_layout()

    return fig, ax


# ============================================================
# PLOT MULTIPLE .NPZ FILES
# ============================================================


def plot_multiple_spreads(
    npz_paths,
    labels=None,
    bins=None,
    offsets=None,
    style=None,
    axis_options=None,
    plot_options=None,
    flux_scales=None,
):
    """Plot and compare one or more spread ``.npz`` files.

    By default all cases are overlaid on a single axis.  Pass
    ``SpreadPlotOptions(layout="side_by_side")`` to create one panel per case.

    ``flux_scales`` and ``offsets`` are applied to every known flux array
    before plotting (model curves and observational data alike):

    .. code-block:: text

        plotted_flux = original_flux * flux_scales[i] + offsets[i]

    This is useful for common-flux normalisation or artificial vertical
    separation between cases.

    Parameters
    ----------
    npz_paths : sequence of str or Path
        Spread cache paths, one per case.
    labels : list[str] or None, default None
        Case labels used as legend prefixes in overlay mode and as panel
        titles in side-by-side mode.  Defaults to the string form of each
        path.
    bins : None, int, or array-like, default None
        Wavelength binning shared by all cases.
    offsets : list[float] or None, default None
        Additive flux offset per case.  Defaults to ``0.0`` for every case.
    style : SpreadPlotStyle or None, default None
        Style options.  Fields may be scalar (shared) or a list with one
        value per case.  Defaults to ``SpreadPlotStyle()``.
    axis_options : AxisOptions or None, default None
        Axis and figure options shared by all panels.
        Defaults to ``AxisOptions()``.
    plot_options : SpreadPlotOptions or None, default None
        Visibility switches and layout selection.
        Defaults to ``SpreadPlotOptions()``.
    flux_scales : list[float] or None, default None
        Multiplicative flux scale per case.  Defaults to ``1.0`` for every
        case.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        ``(fig, ax)`` for overlay layout.
    tuple[matplotlib.figure.Figure, numpy.ndarray of Axes]
        ``(fig, axes)`` for side-by-side layout.

    Raises
    ------
    ValueError
        If ``npz_paths`` is empty, if ``layout`` is not recognised, or if
        any sequence parameter has the wrong length.

    Examples
    --------
    ::

        fig, ax = plot_multiple_spreads(
            ["run1.npz", "run2.npz", "run3.npz"],
            labels=["0.5e-4", "1.0e-4", "1.5e-4"],
            style=SpreadPlotStyle(model_color=["tab:blue", "tab:orange", "tab:green"]),
        )
        display(fig)
    """
    style = style or SpreadPlotStyle()
    plot_options = plot_options or SpreadPlotOptions()
    axis_options = axis_options or AxisOptions()

    n_cases = len(npz_paths)
    if n_cases == 0:
        raise ValueError("npz_paths must contain at least one path.")

    if plot_options.layout not in ("overlay", "side_by_side"):
        raise ValueError("layout must be 'overlay' or 'side_by_side'.")

    if labels is None:
        labels = [str(path) for path in npz_paths]

    if flux_scales is None:
        flux_scales = [1.0] * n_cases
    if offsets is None:
        offsets = [0.0] * n_cases

    for name, seq in [("labels", labels), ("flux_scales", flux_scales), ("offsets", offsets)]:
        if len(seq) != n_cases:
            raise ValueError(f"Expected {n_cases} {name}, got {len(seq)}.")

    if plot_options.layout == "overlay" and style.label_prefix == "":
        style = replace(style, label_prefix=labels)

    case_style_values = _resolve_case_style_values(style, n_cases)

    items_by_case = [_load_spread_items_from_npz(npz_paths[i], case_label=labels[i])
                     for i in range(n_cases)]

    for i, items in enumerate(items_by_case):
        print(f"Loaded {len(items)} observations from: {npz_paths[i]}")

    scaled_items_by_case = [
        [_scale_fluxes(item, float(flux_scales[i]), float(offsets[i])) for item in items]
        for i, items in enumerate(items_by_case)
    ]

    with plt.rc_context(_MPL_RC):
        if plot_options.layout == "side_by_side":
            shared_x_lim = axis_options.x_lim or (
                min(np.nanmin(item["wav"]) for case_items in items_by_case for item in case_items),
                max(np.nanmax(item["wav"]) for case_items in items_by_case for item in case_items),
            )
            x_lims = [shared_x_lim] * n_cases
            figsize = (axis_options.figsize_per_panel[0] * n_cases,
                       axis_options.figsize_per_panel[1])

            fig, axes = plt.subplots(1, n_cases, figsize=figsize, sharey=False,
                                     squeeze=False, constrained_layout=True)
            axes = axes.ravel()  # Always return a flat iterable, including the n_cases=1 case.

            for i, ax in enumerate(axes):
                # Per-panel y_lim: accept a list of per-case y_lims or one shared value.
                y_lim = (axis_options.y_lim[i]
                         if isinstance(axis_options.y_lim, list)
                         and len(axis_options.y_lim) == n_cases
                         else axis_options.y_lim)

                case_axis_options = replace(axis_options, x_lim=x_lims[i], y_lim=y_lim)
                _plot_spread_case(ax, scaled_items_by_case[i], bins,
                                  replace(style, **{f: v[i] for f, v in case_style_values.items()}),
                                  plot_options, case_axis_options)
                ax.set_title(labels[i], fontsize=axis_options.title_size)
                _format_spread_axis(ax, bins, replace(case_axis_options,
                                                      ylabel=axis_options.ylabel if i == 0 else None))

            return fig, axes

        else:
            fig, ax = plt.subplots(figsize=axis_options.figsize, constrained_layout=True)
            data_label_used = False

            for i, case_items in enumerate(scaled_items_by_case):
                data_label_used = _plot_spread_case(
                    ax, case_items, bins,
                    replace(style, **{f: v[i] for f, v in case_style_values.items()}),
                    replace(plot_options, show_sigma_labels=plot_options.show_sigma_labels and i == 0),
                    axis_options, data_label_used,
                )

            _format_spread_axis(ax, bins, axis_options)
            return fig, ax

# ============================================================
# PARAMETER VARIATION
# ============================================================


@dataclass
class InitialModelVariationStyle:
    """
    Visual style for the initial-model variation plots.

    Attributes
    ----------
    plot_observation : bool
        If True, also plot the observed spectrum stored in the cache.
    show_title : bool
        If True, add titles with the initial and fitted parameter values.
    observation_color : str
        Color used for the observed spectrum.
    envelope_color : str | None
        Color used for the filled envelope. If None, each varied parameter uses
        the next color from Matplotlib's default color cycle.
    initial_model_color : str | None
        Color used for the initial model. If None, black is used.
    observation_linewidth : float
        Line width of the observed spectrum.
    difference_linewidth : float
        Line width of the spectrum where one parameter was changed.
    initial_model_linewidth : float
        Line width of the initial-model spectrum.
    difference_linestyle : str
        Line style of the one-parameter-changed spectrum.
    initial_model_linestyle : str
        Line style of the initial-model spectrum.
    envelope_alpha : float
        Opacity of the filled region between the initial model and the
        one-parameter-changed model.
    grid_alpha : float | None
        Grid opacity. Set to None or <= 0 to hide the grid.
    right_label_pad : int
        Padding of the right-side labels in the stacked plot.
    right_difference_x : float
        Horizontal position, in right-axis coordinates, of the numerical
        parameter differences in the stacked plot.
    """

    plot_observation: bool = False
    show_title: bool = True

    observation_color: str = "tab:blue"
    envelope_color: str | None = None
    initial_model_color: str | None = None

    observation_linewidth: float = 0.9
    difference_linewidth: float = 0.8
    initial_model_linewidth: float = 0.7

    difference_linestyle: str = "-"
    initial_model_linestyle: str = "-"

    envelope_alpha: float = 1.0
    grid_alpha: float | None = 0.25

    right_label_pad: int = 8
    right_difference_x: float = 1.08

def _select_parameters(parameters, requested):
    """
    Return the indices of the free parameters that should be varied.

    Parameters
    ----------
    parameters : sequence
        Free parameter objects from ForMoSA. Each parameter is expected to have
        both a ``title`` and a ``name`` attribute.
    requested : sequence[str] | None
        Parameter titles or names requested by the user. If None, all free
        parameters are selected.

    Returns
    -------
    list[int]
        Indices of the selected parameters, in the original ForMoSA free
        parameter order.

    Raises
    ------
    ValueError
        If at least one requested title/name does not match any free parameter.
    """

    if requested is None:
        return list(range(len(parameters)))

    requested = set(requested)
    indices = [
        i
        for i, p in enumerate(parameters)
        if p.title in requested or p.name in requested
    ]
    found = {key for i in indices for key in (parameters[i].title, parameters[i].name)}
    missing = requested - found

    if missing:
        raise ValueError(
            "Requested parameters not found: "
            + ", ".join(sorted(missing))
            + f"\nAvailable titles: {[p.title for p in parameters]}"
            + f"\nAvailable names:  {[p.name for p in parameters]}"
        )

    return indices


def _theta_from_mapping(parameters, values, label):
    """
    Convert a parameter-value dictionary into a theta vector.

    The output order follows ``parameters`` exactly, which is the order expected
    by ``build_models_from_theta``. Each value can be provided using either the
    ForMoSA parameter title or the ForMoSA parameter name.

    Parameters
    ----------
    parameters : sequence
        Free parameter objects from ForMoSA.
    values : mapping
        Dictionary-like object containing one value per free parameter.
    label : str
        Human-readable label used only in error messages.

    Returns
    -------
    numpy.ndarray
        One-dimensional float theta vector.

    Raises
    ------
    ValueError
        If the mapping does not contain a value for every free parameter.
    """

    theta = []
    missing = []

    for parameter in parameters:
        for key in (parameter.title, parameter.name):
            if key in values:
                theta.append(values[key])
                break
        else:
            missing.append(f"{parameter.title} / {parameter.name}")

    if missing:
        raise ValueError(
            f"Missing {label} values for free parameters: "
            + ", ".join(missing)
            + f"\nAvailable keys: {list(values.keys())}"
        )

    return np.asarray(theta, dtype=float)


def _latex_parameter_name(name):
    """
    Return a compact LaTeX label for a ForMoSA parameter name or title.

    Unknown names are still rendered in math mode so that labels remain visually
    consistent with the known parameters.
    """

    names = {
        "Teff": r"$T_{\rm eff}$",
        "log(g)": r"$\log(g)$",
        "[M/H]": r"$[{\rm M/H}]$",
        "C/O": r"${\rm C/O}$",
        "rv": r"${\rm RV}$",
        "vsini": r"$v\sin i$",
        "fsed": r"$f_{\rm sed}$",
    }

    return names.get(str(name), rf"${name}$")


def build_variation_cache(
    config_path,
    output_path,
    initial_model_values,
    obs_index=0,
    parameters_to_vary=None,
    adapted=True,
    fitted=True,
    log_level="warning",
    overwrite=False,
):
    """Build the ``.npz`` variation cache used by the plotting functions.

    For each selected free parameter, evaluates two forward models:

    - the *initial model*, built from ``initial_model_values``;
    - the *comparison model*, where only that parameter is moved to its
      fitted median value while all others are held at their initial values.

    The per-pixel minimum and maximum of the two spectra are saved as
    ``band_low`` and ``band_high``, along with the difference spectrum.
    The resulting envelope is a sensitivity band, not a posterior uncertainty.

    When ``overwrite=False`` and ``output_path`` already exists, the cache is
    reused and no ForMoSA models are rebuilt.

    Parameters
    ----------
    config_path : str or Path
        Path to the ForMoSA ``.ini`` configuration file.
    output_path : str or Path
        Destination path for the ``.npz`` cache.
    initial_model_values : mapping
        Initial values for every free parameter.  Keys may be parameter
        titles or parameter names as defined in ForMoSA.
    obs_index : int, default 0
        Index of the observation to store when ForMoSA builds one model per
        observation.
    parameters_to_vary : sequence[str] or None, default None
        Parameter titles or names to vary.  ``None`` varies all free
        parameters.
    adapted : bool, default True
        Whether to load adapted observations (passed to ``Analysis``).
    fitted : bool, default True
        Whether to load fitted results (passed to ``Analysis``).
    log_level : str, default ``"warning"``
        Logging verbosity passed to ``ConfigLoader`` and ``Analysis``.
    overwrite : bool, default False
        If ``True``, rebuild the cache even if ``output_path`` exists.

    Returns
    -------
    pathlib.Path
        Absolute path to the written or reused ``.npz`` cache file.

    Raises
    ------
    ValueError
        If ``initial_model_values`` does not cover all free parameters, or
        if ``parameters_to_vary`` contains unrecognised names.

    Examples
    --------
    ::

        from spread_plotting import build_variation_cache, plot_variation_stacked

        cache = build_variation_cache(
            config_path="config.ini",
            output_path="variation_cache.npz",
            initial_model_values={"Teff": 1800, "log(g)": 4.0, "C/O": 0.55},
            parameters_to_vary=["Teff", "log(g)"],
        )
        fig, ax, ax_right = plot_variation_stacked(cache)
        display(fig)
    """

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return output_path

    from ForMoSA.analysis import Analysis
    from ForMoSA.config.global_config import ConfigLoader

    config = ConfigLoader(str(config_path), log_level=log_level).load()

    analysis = Analysis(
        config["config_path"],
        adapted=adapted,
        fitted=fitted,
        log_level=log_level,
    )

    ns = analysis.ns
    parameters = list(ns.parameters.free_parameters)
    parameter_titles = np.asarray([p.title for p in parameters], dtype=object)
    parameter_names = np.asarray([p.name for p in parameters], dtype=object)
    varied_indices = np.asarray(
        _select_parameters(parameters, parameters_to_vary),
        dtype=int,
    )

    initial_model_theta = _theta_from_mapping(
        parameters,
        initial_model_values,
        "initial model",
    )
    best_fit_theta = _theta_from_mapping(
        parameters,
        ns.results.median_parameters,
        "best-fit",
    )
    delta_theta = initial_model_theta - best_fit_theta

    observation = ns.restricted_observations.observations[obs_index]
    obs_wave = np.asarray(observation.wave, dtype=float)
    obs_flux = np.asarray(observation.flux, dtype=float)

    def build(theta):
        return analysis.ns_analysis.build_models_from_theta(theta)[obs_index]

    initial_model = build(initial_model_theta)
    wave = np.asarray(initial_model.wave, dtype=float)
    initial_model_flux = np.asarray(initial_model.total_flux, dtype=float)

    low_fluxes = []
    high_fluxes = []
    difference_fluxes = []

    for parameter_index in varied_indices:
        delta = delta_theta[parameter_index]
        difference_theta = initial_model_theta.copy()
        difference_theta[parameter_index] -= delta

        difference_model = build(difference_theta)
        difference_wave = np.asarray(difference_model.wave, dtype=float)
        difference_flux = np.asarray(difference_model.total_flux, dtype=float)

        if difference_wave.shape != wave.shape or not np.allclose(difference_wave, wave):
            difference_flux = np.interp(
                wave,
                difference_wave,
                difference_flux,
                left=np.nan,
                right=np.nan,
            )

        pair = np.vstack([initial_model_flux, difference_flux])
        low_fluxes.append(np.nanmin(pair, axis=0))
        high_fluxes.append(np.nanmax(pair, axis=0))
        difference_fluxes.append(difference_flux)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        config_path=str(config_path),
        obs_index=int(obs_index),
        parameter_titles=parameter_titles,
        parameter_names=parameter_names,
        varied_indices=varied_indices,
        varied_titles=parameter_titles[varied_indices],
        initial_model_theta=initial_model_theta,
        best_fit_theta=best_fit_theta,
        delta_theta=delta_theta,
        wave=wave,
        initial_model_flux=initial_model_flux,
        obs_wave=obs_wave,
        obs_flux=obs_flux,
        band_low=np.asarray(low_fluxes, dtype=float),
        band_high=np.asarray(high_fluxes, dtype=float),
        difference_flux=np.asarray(difference_fluxes, dtype=float),
    )

    return output_path


def plot_variation_by_parameter(
    npz_path,
    n_wavelength_panels=5,
    style=None,
    axis_options=None,
):
    """Plot one figure per varied parameter, split into wavelength panels.

    Each figure shows, for one free parameter:

    - the initial-model spectrum;
    - the spectrum where only that parameter was moved to its fitted value;
    - the filled sensitivity envelope between the two;
    - optionally, the observed spectrum.

    The wavelength axis is divided into ``n_wavelength_panels`` contiguous
    sub-panels to keep narrow spectral features visible.

    Parameters
    ----------
    npz_path : str or Path
        Cache produced by :func:`build_variation_cache`.
    n_wavelength_panels : int, default 5
        Number of wavelength sub-panels per figure.
    style : InitialModelVariationStyle or None, default None
        Visual style for the variation plots.  Defaults to
        ``InitialModelVariationStyle()``.
    axis_options : AxisOptions or None, default None
        Axis and figure options.  Defaults to settings suitable for the
        split-panel layout.

    Returns
    -------
    list[tuple[matplotlib.figure.Figure, numpy.ndarray]]
        One ``(fig, axes)`` pair per varied parameter, in the order they
        appear in the cache.
    """

    if style is None:
        style = InitialModelVariationStyle()

    if axis_options is None:
        axis_options = AxisOptions(
            xlabel="Wavelength",
            ylabel="Initial Model and Difference with Best Fit",
            figsize_per_panel=(11, 2.2),
        )
    elif axis_options.figsize_per_panel is None:
        axis_options = replace(axis_options, figsize_per_panel=(11, 2.2))

    figsize_per_panel = axis_options.figsize_per_panel
    if figsize_per_panel is None:
        raise ValueError("axis_options.figsize_per_panel cannot be None here.")

    data = np.load(npz_path, allow_pickle=True)
    cache = {key: data[key] for key in data.files}
    wave = cache["wave"]
    obs_wave = cache["obs_wave"]

    x_lim = axis_options.x_lim
    if x_lim is None:
        x_lim = (float(np.nanmin(wave)), float(np.nanmax(wave)))

    edges = np.linspace(x_lim[0], x_lim[1], int(n_wavelength_panels) + 1)
    figures = []
    default_colors = _saturate_colors(
        plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
    )

    for row, parameter_index in enumerate(cache["varied_indices"]):
        variable_color = default_colors[row % len(default_colors)]
        band_color = variable_color if style.envelope_color is None else style.envelope_color
        line_color = "black" if style.initial_model_color is None else style.initial_model_color
        title = str(cache["parameter_titles"][parameter_index])
        initial_model_value = cache["initial_model_theta"][parameter_index]
        best_fit = cache["best_fit_theta"][parameter_index]
        delta = cache["delta_theta"][parameter_index]

        fig, axes = plt.subplots(
            int(n_wavelength_panels),
            1,
            figsize=(
                figsize_per_panel[0],
                figsize_per_panel[1] * int(n_wavelength_panels),
            ),
            sharey=True,
            squeeze=False,
        )
        axes = axes.ravel()

        for panel, ax in enumerate(axes):
            left, right = edges[panel], edges[panel + 1]
            include_right = panel == int(n_wavelength_panels) - 1

            model_mask = (wave >= left) & (
                wave <= right if include_right else wave < right
            )
            obs_mask = (obs_wave >= left) & (
                obs_wave <= right if include_right else obs_wave < right
            )

            if style.plot_observation:
                ax.plot(
                    obs_wave[obs_mask],
                    cache["obs_flux"][obs_mask],
                    color=style.observation_color,
                    lw=style.observation_linewidth,
                )

            ax.fill_between(
                wave[model_mask],
                cache["band_low"][row, model_mask],
                cache["band_high"][row, model_mask],
                color=band_color,
                alpha=style.envelope_alpha,
                linewidth=0,
            )

            if "difference_flux" in cache:
                ax.plot(
                    wave[model_mask],
                    cache["difference_flux"][row, model_mask],
                    color=band_color,
                    lw=style.difference_linewidth,
                    ls=style.difference_linestyle,
                )

            ax.plot(
                wave[model_mask],
                cache["initial_model_flux"][model_mask],
                color=line_color,
                lw=style.initial_model_linewidth,
                ls=style.initial_model_linestyle,
            )

            # Axis formatting
            ax.set_xlim(left, right)
            if axis_options.y_lim is not None:
                ax.set_ylim(*axis_options.y_lim)
            if axis_options.ylabel is not None:
                ax.set_ylabel(axis_options.ylabel)
            if style.grid_alpha is not None and style.grid_alpha > 0:
                ax.grid(alpha=style.grid_alpha)
            if style.show_title and panel == 0:
                ax.set_title(
                    f"{_latex_parameter_name(title)}: initial model={initial_model_value:.4g}, "
                    f"best fit={best_fit:.4g}, "
                    rf"$\Delta$={delta:.4g}"
                )
            if include_right and axis_options.xlabel is not None:
                ax.set_xlabel(axis_options.xlabel)

        fig.tight_layout()
        figures.append((fig, axes))

    return figures


def plot_variation_stacked(
    npz_path,
    spacing_factor=1.5,
    style=None,
    axis_options=None,
):
    """Plot all parameter variations in a single vertically stacked figure.

    Each row corresponds to one varied parameter.  Rows are offset vertically
    so they do not overlap.  Right-side axis labels identify the varied
    parameter, and a coloured value in parentheses shows the signed change
    from the initial to the fitted value.

    Parameters
    ----------
    npz_path : str or Path
        Cache produced by :func:`build_variation_cache`.
    spacing_factor : float, default 1.5
        Vertical distance between rows as a multiple of the first row's
        flux range.  Increase to add more whitespace between rows.
    style : InitialModelVariationStyle or None, default None
        Visual style.  Defaults to ``InitialModelVariationStyle()``.
    axis_options : AxisOptions or None, default None
        Axis and figure options.  Defaults to settings suitable for the
        stacked layout.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes]
        The figure, the main (left) axis, and the right-side label axis.
    """

    if style is None:
        style = InitialModelVariationStyle()

    if axis_options is None:
        axis_options = AxisOptions(
            xlabel="Wavelength",
            ylabel="Initial Model and Difference with Best Fit",
            figsize=None,
        )

    data = np.load(npz_path, allow_pickle=True)
    cache = {key: data[key] for key in data.files}
    wave = cache["wave"]
    mask = np.isfinite(wave) if axis_options.x_lim is None else (
        (wave >= axis_options.x_lim[0]) & (wave <= axis_options.x_lim[1])
    )

    x = wave[mask]

    if x.size == 0:
        raise ValueError(
            f"No wavelength points found inside x_lim={axis_options.x_lim}."
        )

    initial_model_flux = cache["initial_model_flux"][mask]
    low = cache["band_low"][:, mask]
    high = cache["band_high"][:, mask]

    difference_flux = None
    if "difference_flux" in cache:
        difference_flux = cache["difference_flux"][:, mask]

    obs = None
    if style.plot_observation:
        obs = np.interp(
            x,
            cache["obs_wave"],
            cache["obs_flux"],
            left=np.nan,
            right=np.nan,
        )

    n_rows = low.shape[0]
    spans = []
    mins = []

    for row in range(n_rows):
        rows = [initial_model_flux, low[row], high[row]]

        if obs is not None:
            rows.append(obs)

        values = np.vstack(rows)
        mins.append(np.nanmin(values))
        spans.append(np.nanmax(values) - mins[-1])

    mins = np.asarray(mins, dtype=float)
    spans = np.asarray(spans, dtype=float)
    reference_span = spans[0] if np.isfinite(spans[0]) and spans[0] > 0 else 1.0
    min_to_min_spacing = float(spacing_factor) * reference_span

    if axis_options.figsize is None:
        axis_options = replace(axis_options, figsize=(12, 1.8 + 1.4 * n_rows))

    fig, ax = plt.subplots(figsize=axis_options.figsize)

    right_positions = []
    right_labels = []
    right_label_differences = []
    default_colors = _saturate_colors(
        plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
    )

    for row in range(n_rows):
        variable_color = default_colors[row % len(default_colors)]
        band_color = variable_color if style.envelope_color is None else style.envelope_color
        line_color = "black" if style.initial_model_color is None else style.initial_model_color
        offset = row * min_to_min_spacing - mins[row]

        if obs is not None:
            ax.plot(
                x,
                obs + offset,
                color=style.observation_color,
                lw=style.observation_linewidth,
            )

        ax.fill_between(
            x,
            low[row] + offset,
            high[row] + offset,
            color=band_color,
            alpha=style.envelope_alpha,
            linewidth=0,
        )

        if difference_flux is not None:
            ax.plot(
                x,
                difference_flux[row] + offset,
                color=band_color,
                lw=style.difference_linewidth,
                ls=style.difference_linestyle,
            )

        ax.plot(
            x,
            initial_model_flux + offset,
            color=line_color,
            lw=style.initial_model_linewidth,
            ls=style.initial_model_linestyle,
        )

        right_positions.append(row * min_to_min_spacing + 0.4 * spans[row])
        varied_title = str(cache["varied_titles"][row])
        parameter_index = int(cache["varied_indices"][row])
        difference = -float(cache["delta_theta"][parameter_index])

        right_labels.append(r"$\Delta$ " + _latex_parameter_name(varied_title))
        right_label_differences.append((difference, band_color))

    title = ", ".join(
        f"{_latex_parameter_name(name)}={value:.4g}"
        for name, value in zip(cache["parameter_titles"], cache["initial_model_theta"])
    )

    if style.show_title:
        ax.set_title(f"Initial model values: {title}")
    if axis_options.xlabel is not None:
        ax.set_xlabel(axis_options.xlabel)
    if axis_options.ylabel is not None:
        ax.set_ylabel(axis_options.ylabel)
    if axis_options.y_lim is not None:
        ax.set_ylim(*axis_options.y_lim)
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    if style.grid_alpha is not None and style.grid_alpha > 0:
        ax.grid(axis="x", alpha=style.grid_alpha)

    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(right_positions)
    ax_right.set_yticklabels(right_labels)
    ax_right.tick_params(
        axis="y",
        right=False,
        labelright=True,
        length=0,
        pad=style.right_label_pad,
    )

    for position, (difference, color) in zip(
        right_positions,
        right_label_differences,
    ):
        ax_right.text(
            style.right_difference_x,
            position,
            f"({difference:+.4g})",
            color=color,
            transform=ax_right.get_yaxis_transform(),
            va="center",
            ha="left",
            clip_on=False,
        )

    fig.tight_layout()

    return fig, ax, ax_right