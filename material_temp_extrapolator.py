"""
Abaqus material interpolation/extrapolation helper.

This script parses an Abaqus *MATERIAL card, interpolates or extrapolates
elastic and plastic data to a target temperature, and exports updated
material cards plus optional plots.

Inputs:
- Abaqus material card text (MATERIAL_TEXT) containing *ELASTIC and *PLASTIC.
- Target temperature (T_TARGET) for interpolation/extrapolation.

Outputs:
- Plots of E(T) and plastic curves (optional).
- Exported .inp material cards with the target temperature inserted.

Methods (high level):
- M1: local linear interpolation in temperature using the two bracketing points.
- M2: local quadratic interpolation in temperature using the three nearest points.
- M3: scales a reference plastic curve by the yield-stress ratio.

Notes:
- Intended for sensitivity studies only.
- Not a substitute for validated material testing data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 0) USER CONFIGURATION
# =============================================================================

# --- Method selection ---
RUN_METHOD_1 = True
RUN_METHOD_2 = False
RUN_METHOD_3 = False

# --- Paste your Abaqus material card here ---
MATERIAL_TEXT = r"""
*MATERIAL, NAME=TRC103P
*DENSITY
                 1.02E-9,
*EXPANSION, TYPE=ISO
                   8.E-5,
*ELASTIC, TYPE=ISOTROPIC
                   1417.,                     0.35,                      23.
                   1303.,                     0.35,                      40.
                   1133.,                     0.35,                      60.
                    921.,                     0.35,                      80.
                    638.,                     0.35,                     100.
*PLASTIC
                  17.204,                       0.,                      23.
                  18.025,              0.016835273,                      23.
                   19.08,              0.044800673,                      23.
                   20.35,              0.080945474,                      23.
                    23.4,               0.16580391,                      23.
                   30.75,              0.383759226,                      23.
                     43.,              0.662794239,                      23.
                15.82768,                       0.,                      40.
                  16.583,              0.016835273,                      40.
                 17.5536,              0.044800673,                      40.
                  18.722,              0.080945474,                      40.
                  21.528,               0.16580391,                      40.
                   28.29,              0.383759226,                      40.
                   39.56,              0.662794239,                      40.
                 13.7632,                       0.,                      60.
                   14.42,              0.016835273,                      60.
                  15.264,              0.044800673,                      60.
                   16.28,              0.080945474,                      60.
                   18.72,               0.16580391,                      60.
                    24.6,              0.383759226,                      60.
                    34.4,              0.662794239,                      60.
                 11.1826,                       0.,                      80.
                 11.7214,              0.016827376,                      80.
                  12.402,              0.044798235,                      80.
                  13.233,                0.0809369,                      80.
                  15.216,              0.165794403,                      80.
                  19.995,               0.38374715,                      80.
                   27.94,              0.662799606,                      80.
                  7.7418,                       0.,                     100.
                  8.1164,              0.016830522,                     100.
                   8.586,              0.044804193,                     100.
                   9.163,              0.080940603,                     100.
                  10.536,              0.165798818,                     100.
                   13.83,              0.383776661,                     100.
                   19.36,              0.662786491,                     100.
"""

# --- Target temperature (any value) ---
T_TARGET = -30.0

# --- Export / plot controls ---
EXPORT_FIGURES = True
SHOW_FIGURES = True
EXPORT_CARDS = False

# --- Output folder/prefixes ---
OUTPUT_PREFIX = "OUT_"

# --- Method M3 (shape) ---
SCALE_REF_T: str | float = "closest"  # or 23.0, 40.0, etc.

# --- Tolerances ---
EPS_MATCH_TOL = 1e-9

# --- Control eps_p domain when grids have different lengths ---
# "intersection" -> use the common domain of temps used by the method (safer)
# "reference"    -> use eps from the reference curve (closest to T_TARGET), trimmed to common domain
EPS_DOMAIN_MODE = "reference"

# --- How to handle eps_p extrapolation when mapping sigma(eps) ---
# "nan"   -> outside range becomes NaN (then masked) [recommended with truncation]
# "clamp" -> plateau (np.interp default behavior) [produces flatline]
# "linear"-> linear extrapolation using first/last segment slope
EPS_EXTRAP_MODE = "nan"

# --- eps_grid size when using "intersection" (artificial grid) ---
EPS_INTERSECTION_NPTS = 250


# =============================================================================
# 1) DATA STRUCTURES / UTILITIES
# =============================================================================

@dataclass
class ElasticData:
    """Parsed elastic data grouped by temperature."""
    temps: np.ndarray
    E: np.ndarray
    nu: float  # assumed constant


@dataclass
class PlasticData:
    """Parsed plastic data, with original curves and a common grid for utilities."""
    temps: np.ndarray
    eps_grid: np.ndarray          # common eps_p grid (utilities only)
    sigma_mat: np.ndarray         # shape (nT, nEps) (utilities only)
    original_curves: Dict[float, Tuple[np.ndarray, np.ndarray]]  # {T: (eps, sigma)}


def _clean_line(s: str) -> str:
    """Trim whitespace and a trailing comma from a line."""
    return s.strip().rstrip(",")


def _is_keyword(line: str) -> bool:
    """Return True if a line starts with an Abaqus keyword marker."""
    return line.strip().startswith("*")


def _parse_floats_from_csv_line(line: str) -> List[float]:
    """Parse a comma-separated line into a list of floats."""
    parts = [p.strip() for p in _clean_line(line).split(",") if p.strip()]
    return [float(p) for p in parts]


def _enforce_non_decreasing(y: np.ndarray) -> np.ndarray:
    """Force a monotonic non-decreasing array."""
    out = y.copy()
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _safe_positive(y: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Clamp values to a small positive floor."""
    return np.maximum(y, floor)


def _find_bracketing_indices(x_sorted: np.ndarray, x0: float) -> Tuple[int, int, str]:
    """
    Return (i_low, i_high, region) where region in {"below", "inside", "above"}.
    Assumes x_sorted is strictly increasing.
    """
    if x0 <= x_sorted[0]:
        return 0, 1 if len(x_sorted) > 1 else 0, "below"
    if x0 >= x_sorted[-1]:
        return (len(x_sorted) - 2 if len(x_sorted) > 1 else 0), len(x_sorted) - 1, "above"

    i_high = int(np.searchsorted(x_sorted, x0, side="right"))
    i_low = i_high - 1
    return i_low, i_high, "inside"


def piecewise_linear_in_T(T: np.ndarray, y: np.ndarray, T_target: float) -> float:
    """
    Local piecewise linear in temperature:
    - inside range: interpolation between bracketing points
    - outside range: linear extrap using first two / last two points
    """
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(T) != len(y):
        raise ValueError("T and y must have same length.")
    if len(T) < 2:
        return float(y[0])

    i0, i1, _ = _find_bracketing_indices(T, T_target)
    T0, T1 = T[i0], T[i1]
    y0, y1 = y[i0], y[i1]
    if np.isclose(T1, T0):
        return float(y0)
    w = (T_target - T0) / (T1 - T0)
    return float(y0 + w * (y1 - y0))


def local_quadratic_in_T(T: np.ndarray, y: np.ndarray, T_target: float) -> float:
    """
    Local quadratic in temperature (3 nearest points to T_target).
    If fewer than 3 points exist, fall back to local linear.
    """
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(T)
    if n < 3:
        return piecewise_linear_in_T(T, y, T_target)

    idx = np.argsort(np.abs(T - T_target))[:3]
    T3 = T[idx]
    y3 = y[idx]
    p = np.polyfit(T3, y3, deg=2)
    return float(np.polyval(p, T_target))


def _choose_ref_temperature(temps: np.ndarray, rule: str | float, T_target: float) -> float:
    if isinstance(rule, (int, float)):
        return float(rule)
    rule = str(rule).strip().lower()
    if rule == "closest":
        return float(temps[int(np.argmin(np.abs(temps - T_target)))])
    raise ValueError("SCALE_REF_T inválido. Use 'closest' ou um número (ex.: 23.0).")


def interp_sigma_vs_eps(eps_new: np.ndarray, eps: np.ndarray, sig: np.ndarray, mode: str) -> np.ndarray:
    """
    Interpolate sigma(eps) and control behavior outside the domain.
    mode:
      - "nan"   -> NaN outside the range
      - "clamp" -> plateau (extremes)
      - "linear"-> linear extrapolation using the first/last segment slope
    """
    eps = np.asarray(eps, float)
    sig = np.asarray(sig, float)
    eps_new = np.asarray(eps_new, float)

    # Start with NaN outside the domain.
    out = np.interp(eps_new, eps, sig, left=np.nan, right=np.nan)

    mode = str(mode).strip().lower()
    if mode == "nan":
        return out

    if mode == "clamp":
        out2 = out.copy()
        out2 = np.where(np.isnan(out2) & (eps_new < eps[0]), sig[0], out2)
        out2 = np.where(np.isnan(out2) & (eps_new > eps[-1]), sig[-1], out2)
        return out2

    if mode == "linear":
        if len(eps) < 2:
            # No slope information: clamp to the only available value.
            return np.where(np.isnan(out), sig[0], out)

        out2 = out.copy()

        # Below the minimum.
        denom0 = (eps[1] - eps[0])
        m0 = (sig[1] - sig[0]) / denom0 if denom0 != 0 else 0.0
        mask_lo = np.isnan(out2) & (eps_new < eps[0])
        out2[mask_lo] = sig[0] + m0 * (eps_new[mask_lo] - eps[0])

        # Above the maximum.
        denom1 = (eps[-1] - eps[-2])
        m1 = (sig[-1] - sig[-2]) / denom1 if denom1 != 0 else 0.0
        mask_hi = np.isnan(out2) & (eps_new > eps[-1])
        out2[mask_hi] = sig[-1] + m1 * (eps_new[mask_hi] - eps[-1])

        return out2

    raise ValueError("EPS_EXTRAP_MODE inválido. Use: 'nan', 'clamp' ou 'linear'.")


def _domain_common_eps(temps_used: List[float], pl: PlasticData) -> Tuple[float, float]:
    """Return (eps_min, eps_max) for the common domain across the given temperatures."""
    eps_mins = []
    eps_maxs = []
    for T in temps_used:
        eps_i, _ = pl.original_curves[float(T)]
        eps_mins.append(float(np.min(eps_i)))
        eps_maxs.append(float(np.max(eps_i)))
    return max(eps_mins), min(eps_maxs)


def _build_eps_target(pl: PlasticData, temps_used: List[float], T_target: float) -> np.ndarray:
    """
    Build eps_target:
      - trimmed to the common domain of temps_used
      - following EPS_DOMAIN_MODE
    """
    eps_min, eps_max = _domain_common_eps(temps_used, pl)
    if eps_max <= eps_min:
        raise ValueError(
            f"Domínio comum de eps_p inválido (eps_max <= eps_min) para temps_used={temps_used}. "
            "As curvas podem não se sobrepor em eps_p."
        )

    mode = str(EPS_DOMAIN_MODE).strip().lower()
    if mode == "intersection":
        return np.linspace(eps_min, eps_max, int(EPS_INTERSECTION_NPTS), dtype=float)

    if mode == "reference":
        # Reference curve: closest to T_target among the available temperatures.
        Tref = float(pl.temps[int(np.argmin(np.abs(pl.temps - T_target)))])
        eps_ref, _ = pl.original_curves[Tref]
        eps_ref = np.asarray(eps_ref, dtype=float)
        eps_ref = eps_ref[(eps_ref >= eps_min) & (eps_ref <= eps_max)]
        if len(eps_ref) < 2:
            # Fallback to intersection if the reference gets too short.
            return np.linspace(eps_min, eps_max, int(EPS_INTERSECTION_NPTS), dtype=float)
        return eps_ref.copy()

    raise ValueError("EPS_DOMAIN_MODE inválido. Use: 'intersection' ou 'reference'.")


# =============================================================================
# 2) ABAQUS CARD PARSER (preserves text before/after blocks)
# =============================================================================

@dataclass
class MaterialSections:
    """Material card split into header, elastic, plastic, and footer sections."""
    header_lines: List[str]        # everything before *ELASTIC
    elastic_kw_line: str           # *ELASTIC keyword line
    elastic_data_lines: List[str]  # numeric elastic data lines
    middle_lines: List[str]        # between end of elastic and start of plastic
    plastic_kw_line: str           # *PLASTIC keyword line
    plastic_data_lines: List[str]  # numeric plastic data lines
    footer_lines: List[str]        # everything after plastic


def split_material_sections(text: str) -> MaterialSections:
    """Split a material card into header/elastic/middle/plastic/footer sections."""
    lines = [l.rstrip("\n") for l in text.splitlines() if l.strip()]

    elastic_idx = None
    plastic_idx = None
    for i, l in enumerate(lines):
        u = l.strip().upper()
        if u.startswith("*ELASTIC"):
            elastic_idx = i
        if u.startswith("*PLASTIC"):
            plastic_idx = i

    if elastic_idx is None or plastic_idx is None:
        raise ValueError("Não encontrei *ELASTIC e/ou *PLASTIC no MATERIAL_TEXT.")
    if plastic_idx < elastic_idx:
        raise ValueError("*PLASTIC apareceu antes de *ELASTIC (inesperado).")

    header = lines[:elastic_idx]
    elastic_kw = lines[elastic_idx]
    elastic_data = []
    i = elastic_idx + 1
    while i < len(lines) and not _is_keyword(lines[i]):
        elastic_data.append(lines[i])
        i += 1

    middle = lines[i:plastic_idx]
    plastic_kw = lines[plastic_idx]
    plastic_data = []
    j = plastic_idx + 1
    while j < len(lines) and not _is_keyword(lines[j]):
        plastic_data.append(lines[j])
        j += 1

    footer = lines[j:]

    return MaterialSections(
        header_lines=header,
        elastic_kw_line=elastic_kw,
        elastic_data_lines=elastic_data,
        middle_lines=middle,
        plastic_kw_line=plastic_kw,
        plastic_data_lines=plastic_data,
        footer_lines=footer,
    )


def parse_elastic(elastic_data_lines: List[str]) -> ElasticData:
    """Parse *ELASTIC data lines into ElasticData."""
    T_list = []
    E_list = []
    nu_list = []
    for l in elastic_data_lines:
        vals = _parse_floats_from_csv_line(l)
        if len(vals) < 3:
            continue
        E_list.append(vals[0])
        nu_list.append(vals[1])
        T_list.append(vals[2])

    if len(T_list) == 0:
        raise ValueError("Bloco *ELASTIC vazio ou inválido.")

    temps = np.array(T_list, dtype=float)
    E = np.array(E_list, dtype=float)
    nu_arr = np.array(nu_list, dtype=float)

    idx = np.argsort(temps)
    temps = temps[idx]
    E = E[idx]
    nu_arr = nu_arr[idx]

    nu = float(nu_arr[0])
    if np.max(np.abs(nu_arr - nu)) > 1e-6:
        print("[WARN] ν varia com T no input; o script vai assumir ν = primeiro valor.")

    return ElasticData(temps=temps, E=E, nu=nu)


def parse_plastic(plastic_data_lines: List[str]) -> PlasticData:
    """Parse *PLASTIC data lines into PlasticData."""
    curves: Dict[float, List[Tuple[float, float]]] = {}
    for l in plastic_data_lines:
        vals = _parse_floats_from_csv_line(l)
        if len(vals) < 3:
            continue
        sigma, eps_p, T = float(vals[0]), float(vals[1]), float(vals[2])
        curves.setdefault(T, []).append((eps_p, sigma))

    if len(curves) == 0:
        raise ValueError("Bloco *PLASTIC vazio ou inválido.")

    original: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for T, pts in curves.items():
        pts = sorted(pts, key=lambda x: x[0])
        eps = np.array([p[0] for p in pts], dtype=float)
        sig = np.array([p[1] for p in pts], dtype=float)
        original[float(T)] = (eps, sig)

    temps = np.array(sorted(original.keys()), dtype=float)

    # Build a "global" eps_grid only for utilities (M3 uses it for sigma_y).
    eps_base = original[float(temps[0])][0]
    same_grid = True
    for T in temps[1:]:
        eps_i = original[float(T)][0]
        if len(eps_i) != len(eps_base) or np.max(np.abs(eps_i - eps_base)) > EPS_MATCH_TOL:
            same_grid = False
            break

    if same_grid:
        eps_grid = eps_base.copy()
    else:
        # Global grid = union of all eps to enable stable interpolation and sigma_y(T).
        eps_all = np.concatenate([original[float(T)][0] for T in temps])
        eps_grid = np.unique(np.sort(eps_all))

    sigma_mat = np.zeros((len(temps), len(eps_grid)), dtype=float)
    for i, T in enumerate(temps):
        eps_i, sig_i = original[float(T)]
        # For the global grid, use clamp (np.interp) only to populate sigma_mat.
        # This sigma_mat is NOT used to generate M1/M2 (to avoid flatline).
        sigma_mat[i, :] = np.interp(eps_grid, eps_i, sig_i)

    for i in range(len(temps)):
        sigma_mat[i, :] = _enforce_non_decreasing(_safe_positive(sigma_mat[i, :]))

    return PlasticData(temps=temps, eps_grid=eps_grid, sigma_mat=sigma_mat, original_curves=original)

# =============================================================================
# 3) COMPUTE PROPERTIES AT T_TARGET (3 METHODS)
# =============================================================================

@dataclass
class MethodResult:
    """Result of a single extrapolation/interpolation method."""
    tag: str
    E_target: float
    nu: float
    eps_target: np.ndarray
    sigma_target: np.ndarray


def compute_target_linear(el: ElasticData, pl: PlasticData, T_target: float) -> MethodResult:
    """
    M1: local linear in T using ONLY the two temperatures bracketing T_target.
    eps_target is trimmed to the common domain of those curves to avoid flatline.
    """
    E_t = piecewise_linear_in_T(el.temps, el.E, T_target)

    i0, i1, _ = _find_bracketing_indices(pl.temps, T_target)
    T0 = float(pl.temps[i0])
    T1 = float(pl.temps[i1])

    temps_used = [T0, T1]
    eps_target = _build_eps_target(pl, temps_used, T_target)

    eps0, sig0 = pl.original_curves[T0]
    eps1, sig1 = pl.original_curves[T1]

    s0 = interp_sigma_vs_eps(eps_target, eps0, sig0, mode=EPS_EXTRAP_MODE)
    s1 = interp_sigma_vs_eps(eps_target, eps1, sig1, mode=EPS_EXTRAP_MODE)

    mask = ~np.isnan(s0) & ~np.isnan(s1)
    eps_target = eps_target[mask]
    s0 = s0[mask]
    s1 = s1[mask]

    if np.isclose(T1, T0):
        sigma_t = s0
    else:
        w = (T_target - T0) / (T1 - T0)
        sigma_t = s0 + w * (s1 - s0)

    sigma_t = _enforce_non_decreasing(_safe_positive(sigma_t))

    return MethodResult(
        tag="M1_LINEAR_LOCAL",
        E_target=float(max(E_t, 1e-12)),
        nu=el.nu,
        eps_target=eps_target,
        sigma_target=sigma_t,
    )


def compute_target_local_quadratic(el: ElasticData, pl: PlasticData, T_target: float) -> MethodResult:
    """
    M2: local quadratic in T using the three nearest temperatures to T_target.
    eps_target is trimmed to the common domain of those curves to avoid flatline.
    """
    E_t = local_quadratic_in_T(el.temps, el.E, T_target)

    if len(pl.temps) < 3:
        # Fallback directly to linear.
        fallback = compute_target_linear(el, pl, T_target)
        return MethodResult(
            tag="M2_QUADRATIC_LOCAL_FALLBACK",
            E_target=fallback.E_target,
            nu=fallback.nu,
            eps_target=fallback.eps_target,
            sigma_target=fallback.sigma_target,
        )

    idx3 = np.argsort(np.abs(pl.temps - T_target))[:3]
    T_used = [float(pl.temps[i]) for i in idx3]

    eps_target = _build_eps_target(pl, T_used, T_target)

    # Compute sigma(eps) at three temperatures and fit a quadratic in T point-by-point.
    sig_stack = []
    for T in T_used:
        eps_i, sig_i = pl.original_curves[float(T)]
        s_i = interp_sigma_vs_eps(eps_target, eps_i, sig_i, mode=EPS_EXTRAP_MODE)
        sig_stack.append(s_i)

    sig_stack = np.vstack(sig_stack)  # (3, nEps)
    # Mask points where any temperature returned NaN (should not happen with truncation).
    mask = np.all(~np.isnan(sig_stack), axis=0)
    eps_target = eps_target[mask]
    sig_stack = sig_stack[:, mask]

    # Quadratic fit in T for each column.
    T3 = np.array(T_used, dtype=float)
    sigma_t = np.zeros(sig_stack.shape[1], dtype=float)
    for j in range(sig_stack.shape[1]):
        y3 = sig_stack[:, j]
        p = np.polyfit(T3, y3, deg=2)
        sigma_t[j] = float(np.polyval(p, T_target))

    sigma_t = _enforce_non_decreasing(_safe_positive(sigma_t))

    return MethodResult(
        tag="M2_QUADRATIC_LOCAL",
        E_target=float(max(E_t, 1e-12)),
        nu=el.nu,
        eps_target=eps_target,
        sigma_target=sigma_t,
    )


def compute_target_scaled_by_yield(el: ElasticData, pl: PlasticData, T_target: float, ref_rule: str | float) -> MethodResult:
    """
    M3: scale a reference curve by the ratio sigma_y(T_target)/sigma_y(T_ref).
    No tail is created because eps_ref from the base curve is used.
    """
    E_t = piecewise_linear_in_T(el.temps, el.E, T_target)

    T_ref = _choose_ref_temperature(pl.temps, ref_rule, T_target)
    if T_ref not in pl.original_curves:
        T_ref = float(pl.temps[int(np.argmin(np.abs(pl.temps - T_ref)))])

    eps_ref, sig_ref = pl.original_curves[float(T_ref)]

    idx0 = int(np.argmin(np.abs(pl.eps_grid - 0.0)))
    sigma_y_T = pl.sigma_mat[:, idx0]
    sigma_y_target = piecewise_linear_in_T(pl.temps, sigma_y_T, T_target)

    idx0_ref = int(np.argmin(np.abs(eps_ref - 0.0)))
    sigma_y_ref = float(sig_ref[idx0_ref])
    if sigma_y_ref <= 0:
        raise ValueError("σy_ref <= 0 (dados de referência estranhos).")

    scale = float(sigma_y_target / sigma_y_ref)

    sig_t = sig_ref * scale
    sig_t = _enforce_non_decreasing(_safe_positive(sig_t))

    return MethodResult(
        tag=f"M3_SCALE_BY_YIELD_ref{T_ref:g}",
        E_target=float(max(E_t, 1e-12)),
        nu=el.nu,
        eps_target=np.asarray(eps_ref, dtype=float).copy(),
        sigma_target=np.asarray(sig_t, dtype=float).copy(),
    )


# =============================================================================
# 4) PLOTS (E(T) AND PLASTIC CURVES)
# =============================================================================

def plot_elastic(
    el: ElasticData,
    T_target: float,
    results: List[MethodResult],
    export: bool,
    show: bool,
    filename: str,
):
    """Plot E(T) and highlight the target temperature for each method."""
    plt.figure()
    plt.plot(
        el.temps,
        el.E,
        marker="o",
        linestyle="-",
        color="0.15",
        markerfacecolor="0.15",
        markeredgecolor="0.15",
        label="Original E(T)",
    )
    for res in results:
        if res.tag.startswith("M1"):
            marker = "x"
        elif res.tag.startswith("M2"):
            marker = "+"
        elif res.tag.startswith("M3"):
            marker = "^"
        else:
            marker = "x"
        plt.scatter(
            [T_target],
            [res.E_target],
            color="red",
            marker=marker,
            label=f"{res.tag} @ {T_target:g} °C",
        )
    plt.xlabel("Temperatura (°C)")
    plt.ylabel("E (MPa)")
    plt.title("Módulo elástico vs Temperatura")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if export:
        plt.savefig(filename, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_plastic(
    pl: PlasticData,
    T_target: float,
    results: List[MethodResult],
    export: bool,
    show: bool,
    filename: str,
):
    """Plot original plastic curves and the extrapolated/interpolated results."""
    plt.figure()

    if len(pl.temps) > 1:
        colors = plt.cm.Greys(np.linspace(0.25, 0.75, len(pl.temps)))
    else:
        colors = ["0.35"]

    for idx, T in enumerate(pl.temps):
        eps_i, sig_i = pl.original_curves[float(T)]
        plt.plot(eps_i, sig_i, linestyle="-", color=colors[idx], label=f"{T:g} °C")

    for res in results:
        if res.tag.startswith("M1"):
            linestyle = "-"
        elif res.tag.startswith("M2"):
            linestyle = ":"
        elif res.tag.startswith("M3"):
            linestyle = "--"
        else:
            linestyle = "-"
        plt.plot(
            res.eps_target,
            res.sigma_target,
            color="red",
            linestyle=linestyle,
            linewidth=2,
            label=f"{res.tag} @ {T_target:g} °C",
        )

    plt.xlabel("Deformação plástica εp")
    plt.ylabel("Tensão σ (MPa)")
    plt.title(f"Curvas *PLASTIC + (interpolação/extrapolação) em T_TARGET = {T_target:g} °C")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if export:
        plt.savefig(filename, dpi=200)
    if show:
        plt.show()
    plt.close()


# =============================================================================
# 5) EXPORT FULL MATERIAL CARD ORDERED BY TEMPERATURE
# =============================================================================

def _format_elastic_line(E: float, nu: float, T: float) -> str:
    return f"{E:>20.6f}, {nu:>12.6f}, {T:>12.6f}"


def _format_plastic_line(sigma: float, eps_p: float, T: float) -> str:
    return f"{sigma:>20.6f}, {eps_p:>16.9f}, {T:>12.6f}"


def export_full_material_inp(
    sections: MaterialSections,
    el: ElasticData,
    pl: PlasticData,
    T_target: float,
    res: MethodResult,
    out_path: str,
):
    """
    Export a full .inp material card:
    - preserve header, middle, and footer as in the input
    - rewrite *ELASTIC and *PLASTIC with temperatures in ascending order,
      inserting T_target if it does not exist, or replacing its block if it does
    """
    E_map = {float(T): float(E) for T, E in zip(el.temps, el.E)}
    E_map[float(T_target)] = float(res.E_target)

    temps_el = np.array(sorted(E_map.keys()), dtype=float)
    elastic_lines = [_format_elastic_line(E_map[float(T)], el.nu, float(T)) for T in temps_el]

    pl_map: Dict[float, Tuple[np.ndarray, np.ndarray]] = {float(T): pl.original_curves[float(T)] for T in pl.temps}
    pl_map[float(T_target)] = (res.eps_target, res.sigma_target)

    temps_pl = np.array(sorted(pl_map.keys()), dtype=float)

    plastic_lines: List[str] = []
    for T in temps_pl:
        eps_i, sig_i = pl_map[float(T)]
        sig_i = _enforce_non_decreasing(_safe_positive(np.asarray(sig_i, float)))
        eps_i = np.asarray(eps_i, float)
        for s, e in zip(sig_i, eps_i):
            plastic_lines.append(_format_plastic_line(float(s), float(e), float(T)))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("** ============================================================\n")
        f.write("** Auto-generated material card with interpolated/extrapolated T_TARGET\n")
        f.write(f"** T_TARGET = {T_target:g} °C | Method = {res.tag}\n")
        f.write(f"** EPS_DOMAIN_MODE={EPS_DOMAIN_MODE} | EPS_EXTRAP_MODE={EPS_EXTRAP_MODE}\n")
        f.write("** ============================================================\n\n")

        for l in sections.header_lines:
            f.write(l + "\n")

        f.write(sections.elastic_kw_line + "\n")
        for l in elastic_lines:
            f.write(l + "\n")

        for l in sections.middle_lines:
            f.write(l + "\n")

        f.write(sections.plastic_kw_line + "\n")
        for l in plastic_lines:
            f.write(l + "\n")

        for l in sections.footer_lines:
            f.write(l + "\n")

# =============================================================================
# 6) MAIN
# =============================================================================

def main():
    """Run the full parse/compute/plot/export pipeline."""
    sections = split_material_sections(MATERIAL_TEXT)
    el = parse_elastic(sections.elastic_data_lines)
    pl = parse_plastic(sections.plastic_data_lines)

    if not (RUN_METHOD_1 or RUN_METHOD_2 or RUN_METHOD_3):
        raise ValueError(
            "At least one extrapolation method must be enabled: RUN_METHOD_1 / RUN_METHOD_2 / RUN_METHOD_3."
        )

    results: List[MethodResult] = []
    if RUN_METHOD_1:
        results.append(compute_target_linear(el, pl, T_TARGET))
    if RUN_METHOD_2:
        results.append(compute_target_local_quadratic(el, pl, T_TARGET))
    if RUN_METHOD_3:
        results.append(compute_target_scaled_by_yield(el, pl, T_TARGET, SCALE_REF_T))

    figE = f"{OUTPUT_PREFIX}Elastic_E_vs_T_T{T_TARGET:g}.png"
    figP = f"{OUTPUT_PREFIX}Plastic_curves_T{T_TARGET:g}.png"

    plot_elastic(el, T_TARGET, results, EXPORT_FIGURES, SHOW_FIGURES, figE)
    plot_plastic(pl, T_TARGET, results, EXPORT_FIGURES, SHOW_FIGURES, figP)

    if EXPORT_CARDS:
        mat_name = "MATERIAL"
        m = re.search(r"\*MATERIAL\s*,\s*NAME\s*=\s*([^\s,]+)", MATERIAL_TEXT, flags=re.IGNORECASE)
        if m:
            mat_name = m.group(1).strip()

        for res in results:
            safe_tag = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", res.tag)
            out_inp = f"{OUTPUT_PREFIX}{mat_name}_T{T_TARGET:g}_{safe_tag}.inp"
            export_full_material_inp(sections, el, pl, T_TARGET, res, out_inp)

    print("OK.")
    if EXPORT_FIGURES:
        print(f"- Figuras: {figE} | {figP}")
    if EXPORT_CARDS:
        print("- Cartas .inp exportadas (uma por método), com sufixo do método no nome do arquivo.")


if __name__ == "__main__":
    main()
