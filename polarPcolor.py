"""Polar pseudocolor plot for gridded data.

This is a Python/Matplotlib version of Etienne Cheynet's MATLAB
``polarPcolor`` function. It plots a matrix ``Z`` on a polar-like grid while
using Cartesian axes internally, which makes it easy to overlay annotations.

Angle conventions
-----------------
``type_rose='meteo'``:
    0 degrees points North and angles increase clockwise.

``type_rose='default'``:
    0 degrees points East and angles increase counter-clockwise.

Expected data layout
--------------------
The internal layout is ``Z.shape == (len(R), len(theta))``. For convenience,
``Z.shape == (len(theta), len(R))`` is also accepted and transposed when the
shape is unambiguous.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar

RoseType = Literal["meteo", "default"]
RadialScale = Literal["linear", "lin", "log", "logarithmic"]


def polar_pcolor(
    R: Sequence[float],
    theta: Sequence[float],
    Z: np.ndarray,
    *,
    ax: Axes | None = None,
    n_circles: int = 5,
    n_spokes: int = 8,
    label_r: str = "",
    r_tick_labels: Sequence[str] | None = None,
    colorbar: bool = True,
    r_scale: RadialScale = "linear",
    auto_origin: bool | str = True,
    cmap: str = "viridis",
    n_colors: int | None = None,
    type_rose: RoseType = "meteo",
    circles_pos: Sequence[float] | None = None,
    robust_limits: bool = True,
    clim_quantiles: tuple[float, float] = (0.01, 0.99),
    vmin: float | None = None,
    vmax: float | None = None,
    shading: str = "flat",
    grid: bool = True,
    grid_color: str | tuple[float, float, float] = "0.5",
    grid_linewidth: float = 0.8,
    angle_label_radius: float = 1.06,
    r_label_angle: float | None = None,
    colorbar_location: str = "left",
    rasterized: bool = False,
) -> tuple[ScalarMappable, Colorbar | None]:
    """Plot ``Z`` on a polar pseudocolor grid.

    Parameters
    ----------
    R : sequence of float
        Radial coordinate, length ``nr``.
    theta : sequence of float
        Angular coordinate in degrees, length ``nt``.
    Z : array-like
        Data matrix. Preferred shape is ``(nr, nt)``. Shape ``(nt, nr)`` is
        accepted and transposed when ``nr != nt``.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw. If omitted, the current axes are used.
    n_circles, n_spokes : int
        Number of radial circles and angular spokes used for the overlaid grid.
        Set either to 0 to hide that grid component.
    label_r : str
        Label for the radial coordinate.
    r_tick_labels : sequence of str, optional
        Custom radial tick labels. Must have the same length as the number of
        circles after ``circles_pos`` has been applied.
    colorbar : bool
        Add a colorbar if True.
    r_scale : {"linear", "lin", "log", "logarithmic"}
        Radial scaling used for display.
    auto_origin : bool or {"on", "off"}
        If True/"on", the first radial value is shown at its own radius, as in
        the original MATLAB function. If False/"off", the first radial value is
        placed at the plot origin. Log scaling always maps ``min(R)`` to the
        display origin because zero is not defined on a logarithmic radial axis.
    cmap : str
        Matplotlib colormap name. ``"parula"`` is mapped to ``"viridis"`` when
        a parula colormap is not registered.
    n_colors : int, optional
        Number of discrete colors used from the colormap.
    type_rose : {"meteo", "default"}
        Angle convention. ``"meteo"`` means 0 degrees is North and angles are
        clockwise. ``"default"`` means 0 degrees is East and angles are
        counter-clockwise.
    circles_pos : sequence of float, optional
        Explicit radial circle positions. Values outside the range of ``R`` are
        ignored. ``min(R)`` and ``max(R)`` are always included.
    robust_limits : bool
        If True and no explicit ``vmin``/``vmax`` is supplied, color limits are
        set to the 1st and 99th percentiles, matching the intent of the MATLAB
        code. NaNs are ignored.
    clim_quantiles : tuple of float
        Quantiles used when ``robust_limits`` is True.
    vmin, vmax : float, optional
        Explicit color limits. These override ``robust_limits``.
    shading : str
        Shading argument passed to ``Axes.pcolormesh``. The default, ``"flat"``,
        uses explicitly computed cell edges and avoids the non-monotonic-coordinate
        warning that can occur with polar meshes.
    grid : bool
        Draw circles, spokes and labels if True.
    grid_color, grid_linewidth : optional
        Appearance of the overlaid polar grid.
    angle_label_radius : float
        Multiplier applied to the outer radius for angular labels.
    r_label_angle : float, optional
        Angular position of radial tick labels, in the same convention as
        ``theta``. If omitted, a position near the middle of the angular sector
        is used.
    colorbar_location : str
        Matplotlib colorbar location. ``"left"`` mimics MATLAB's
        ``WestOutside`` reasonably well.
    rasterized : bool
        Rasterize the pseudocolor mesh when saving vector figures.

    Returns
    -------
    mesh : matplotlib.cm.ScalarMappable
        The pseudocolor mesh returned by ``Axes.pcolormesh``.
    cbar : matplotlib.colorbar.Colorbar or None
        The colorbar, if requested.
    """
    R = _as_1d_float_array(R, "R")
    theta = _as_1d_float_array(theta, "theta")
    Z = _prepare_z(Z, R.size, theta.size)

    if R.size < 2:
        raise ValueError("R must contain at least two values.")
    if theta.size < 2:
        raise ValueError("theta must contain at least two values.")
    if not np.all(np.diff(R) > 0):
        raise ValueError("R must be strictly increasing.")
    if not np.all(np.diff(theta) != 0):
        raise ValueError("theta must not contain repeated values.")

    type_rose = type_rose.lower()
    if type_rose not in {"meteo", "default"}:
        raise ValueError('type_rose must be "meteo" or "default".')

    r_scale = r_scale.lower()
    if r_scale not in {"linear", "lin", "log", "logarithmic"}:
        raise ValueError('r_scale must be "linear"/"lin" or "log"/"logarithmic".')

    auto_origin_bool = _parse_auto_origin(auto_origin)
    if r_scale in {"log", "logarithmic"}:
        if np.nanmin(R) <= 0:
            raise ValueError("R must be strictly positive when r_scale='log'.")
        if not auto_origin_bool:
            warnings.warn(
                "auto_origin=False is not meaningful for log-scaled radius. "
                "The display radius is mapped from log10(R/min(R)).",
                RuntimeWarning,
                stacklevel=2,
            )

    ax = plt.gca() if ax is None else ax

    rho = _radial_transform(R, R, r_scale, auto_origin_bool)
    rho_edges = _centers_to_edges(rho, lower_bound=0.0)
    theta_edges = _centers_to_edges(theta)
    theta_grid, rho_grid = np.meshgrid(theta_edges, rho_edges)
    X, Y = _polar_to_xy(rho_grid, theta_grid, type_rose)

    used_cmap = _get_cmap(cmap, n_colors)
    mesh = ax.pcolormesh(
        X,
        Y,
        Z,
        shading=shading,
        cmap=used_cmap,
        rasterized=rasterized,
    )

    if vmin is not None or vmax is not None:
        mesh.set_clim(vmin=vmin, vmax=vmax)
    elif robust_limits:
        finite = Z[np.isfinite(Z)]
        if finite.size:
            q_low, q_high = np.nanquantile(finite, clim_quantiles)
            if q_low < q_high:
                mesh.set_clim(q_low, q_high)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    circle_values = _circle_values(R, n_circles, circles_pos, r_scale)
    circle_rho = _radial_transform(circle_values, R, r_scale, auto_origin_bool)

    if grid and n_circles > 0 and n_spokes > 0:
        _draw_spokes(
            ax,
            theta_min=float(theta.min()),
            theta_max=float(theta.max()),
            n_spokes=n_spokes,
            circle_rho=circle_rho,
            type_rose=type_rose,
            color=grid_color,
            linewidth=grid_linewidth,
            label_radius_factor=angle_label_radius,
        )
        _draw_circles(
            ax,
            theta_min=float(theta.min()),
            theta_max=float(theta.max()),
            circle_rho=circle_rho,
            type_rose=type_rose,
            color=grid_color,
            linewidth=grid_linewidth,
        )
        _draw_radial_labels(
            ax,
            theta_min=float(theta.min()),
            theta_max=float(theta.max()),
            n_spokes=n_spokes,
            circle_values=circle_values,
            circle_rho=circle_rho,
            r_tick_labels=r_tick_labels,
            label_r=label_r,
            type_rose=type_rose,
            r_label_angle=r_label_angle,
        )

    cbar = None
    if colorbar:
        cbar = plt.colorbar(mesh, ax=ax, location=colorbar_location)

    return mesh, cbar


def _as_1d_float_array(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _prepare_z(Z: np.ndarray, nr: int, nt: int) -> np.ndarray:
    z = np.asarray(Z, dtype=float)
    if z.ndim != 2:
        raise ValueError("Z must be a two-dimensional array.")

    if z.shape == (nr, nt):
        return z
    if z.shape == (nt, nr) and nr != nt:
        return z.T

    raise ValueError(
        "Z has incompatible dimensions. Expected shape "
        f"({nr}, {nt}) or, unambiguously, ({nt}, {nr}); got {z.shape}."
    )


def _parse_auto_origin(auto_origin: bool | str) -> bool:
    if isinstance(auto_origin, str):
        value = auto_origin.strip().lower()
        if value in {"on", "true", "1", "yes"}:
            return True
        if value in {"off", "false", "0", "no"}:
            return False
        raise ValueError('auto_origin must be True/False or "on"/"off".')
    return bool(auto_origin)


def _radial_transform(
    values: np.ndarray,
    R_reference: np.ndarray,
    r_scale: str,
    auto_origin: bool,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    r_min = float(np.min(R_reference))

    if r_scale in {"linear", "lin"}:
        if auto_origin:
            return values.copy()
        return values - r_min

    # Logarithmic display: radius is proportional to log10(R/min(R)).
    if np.any(values <= 0):
        raise ValueError("Radial values must be positive when r_scale='log'.")
    return np.log10(values) - np.log10(r_min)



def _centers_to_edges(values: np.ndarray, lower_bound: float | None = None) -> np.ndarray:
    """Convert one-dimensional cell centers to cell edges."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two center values are required to compute edges.")

    diffs = np.diff(values)
    if np.any(diffs == 0):
        raise ValueError("Center values must not contain repeated entries.")

    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = values[:-1] + 0.5 * diffs
    edges[0] = values[0] - 0.5 * diffs[0]
    edges[-1] = values[-1] + 0.5 * diffs[-1]

    if lower_bound is not None and edges[0] < lower_bound:
        edges[0] = lower_bound

    return edges

def _polar_to_xy(
    rho: np.ndarray,
    theta_deg: np.ndarray,
    type_rose: str,
) -> tuple[np.ndarray, np.ndarray]:
    if type_rose == "meteo":
        phi = np.deg2rad(90.0 - theta_deg)
    elif type_rose == "default":
        phi = np.deg2rad(theta_deg)
    else:
        raise ValueError('type_rose must be "meteo" or "default".')

    return rho * np.cos(phi), rho * np.sin(phi)


def _circle_values(
    R: np.ndarray,
    n_circles: int,
    circles_pos: Sequence[float] | None,
    r_scale: str,
) -> np.ndarray:
    r_min = float(np.min(R))
    r_max = float(np.max(R))

    if circles_pos is not None:
        circles = np.asarray(circles_pos, dtype=float).ravel()
        circles = circles[np.isfinite(circles)]
        circles = circles[(circles >= r_min) & (circles <= r_max)]
        circles = np.unique(np.r_[r_min, circles, r_max])
        return circles

    if n_circles <= 0:
        return np.array([], dtype=float)

    if r_scale in {"linear", "lin"}:
        return np.linspace(r_min, r_max, n_circles)
    return np.logspace(np.log10(r_min), np.log10(r_max), n_circles)


def _get_cmap(cmap: str, n_colors: int | None):
    cmap_name = "viridis" if cmap.lower() == "parula" else cmap
    try:
        return plt.get_cmap(cmap_name, n_colors) if n_colors else plt.get_cmap(cmap_name)
    except ValueError:
        if cmap.lower() == "parula":
            warnings.warn(
                "The parula colormap is not registered in Matplotlib. "
                "Using viridis instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            return plt.get_cmap("viridis", n_colors) if n_colors else plt.get_cmap("viridis")
        raise


def _draw_spokes(
    ax: Axes,
    *,
    theta_min: float,
    theta_max: float,
    n_spokes: int,
    circle_rho: np.ndarray,
    type_rose: str,
    color,
    linewidth: float,
    label_radius_factor: float,
) -> None:
    spoke_angles = np.linspace(theta_min, theta_max, n_spokes)
    full_circle = np.isclose((theta_max - theta_min) % 360.0, 0.0) and not np.isclose(theta_max, theta_min)
    outer_radius = float(np.nanmax(circle_rho))
    inner_radius = float(np.nanmin(circle_rho))

    for angle in spoke_angles:
        x, y = _polar_to_xy(
            np.array([inner_radius, outer_radius]),
            np.array([angle, angle]),
            type_rose,
        )
        ax.plot(x, y, color=color, linewidth=linewidth, zorder=3)

        if full_circle and np.isclose(angle, theta_max):
            continue

        xt, yt = _polar_to_xy(
            np.array([outer_radius * label_radius_factor]),
            np.array([angle]),
            type_rose,
        )
        ax.text(
            xt[0],
            yt[0],
            f"{angle:.1f}\N{DEGREE SIGN}",
            ha="center",
            va="center",
            zorder=4,
        )


def _draw_circles(
    ax: Axes,
    *,
    theta_min: float,
    theta_max: float,
    circle_rho: np.ndarray,
    type_rose: str,
    color,
    linewidth: float,
) -> None:
    angle_grid = np.linspace(theta_min, theta_max, 361)
    for rho in circle_rho:
        rho_grid = np.full_like(angle_grid, rho, dtype=float)
        x, y = _polar_to_xy(rho_grid, angle_grid, type_rose)
        ax.plot(x, y, color=color, linewidth=linewidth, zorder=3)


def _draw_radial_labels(
    ax: Axes,
    *,
    theta_min: float,
    theta_max: float,
    n_spokes: int,
    circle_values: np.ndarray,
    circle_rho: np.ndarray,
    r_tick_labels: Sequence[str] | None,
    label_r: str,
    type_rose: str,
    r_label_angle: float | None,
) -> None:
    if circle_values.size == 0:
        return

    if r_tick_labels is None:
        labels = [_format_tick_label(value) for value in circle_values]
    else:
        labels = list(r_tick_labels)
        if len(labels) != len(circle_values):
            raise ValueError(
                "r_tick_labels must have the same length as the number of radial circles."
            )
        if not all(isinstance(label, str) for label in labels):
            raise TypeError("r_tick_labels must be a sequence of strings.")

    if r_label_angle is None:
        if n_spokes > 1:
            spokes = np.linspace(theta_min, theta_max, n_spokes)
            mid = max(1, min(len(spokes) - 1, len(spokes) // 2))
            r_label_angle = 0.5 * (spokes[mid - 1] + spokes[mid])
        else:
            r_label_angle = theta_min

    for rho, label in zip(circle_rho, labels):
        x, y = _polar_to_xy(np.array([rho]), np.array([r_label_angle]), type_rose)
        ax.text(
            x[0],
            y[0],
            label,
            ha="right",
            va="baseline",
            zorder=4,
        )

    if label_r:
        outer_radius = float(np.nanmax(circle_rho))
        factor = 1.25 if abs(theta_max - theta_min) > 180 else 0.65
        x, y = _polar_to_xy(
            np.array([outer_radius * factor]),
            np.array([r_label_angle]),
            type_rose,
        )
        ax.text(
            x[0],
            y[0],
            label_r,
            ha="right",
            va="bottom",
            zorder=4,
        )


def _format_tick_label(value: float) -> str:
    if np.isclose(value, 0):
        return "0"
    if abs(value) >= 100 or abs(value) < 0.01:
        return f"{value:.2g}"
    return f"{value:.2f}"


if __name__ == "__main__":
    R = np.linspace(3, 10, 100)
    theta = np.linspace(0, 180, 360)
    Z = np.outer(np.linspace(0, 10, theta.size), np.linspace(0, 10, R.size))

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    polar_pcolor(
        R,
        theta,
        Z,
        ax=ax,
        n_circles=3,
        n_spokes=7,
        label_r="r (km)",
        type_rose="meteo",
    )
    fig.savefig("polar_pcolor_demo.png", dpi=200)
