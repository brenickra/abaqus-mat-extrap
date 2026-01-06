# Abaqus Material Temperature Extrapolator

Generate interpolated or extrapolated Abaqus *ELASTIC and *PLASTIC data at a target temperature for sensitivity studies.

## Problem / Motivation
Abaqus material properties are often tabulated versus temperature, but an analysis can require a temperature outside (or between) the available data points. Abaqus does not extrapolate these properties in a physically meaningful way, yet engineers may still need estimated cards for what-if or sensitivity runs.

## What this tool does
- Parses an Abaqus *MATERIAL card and preserves header, middle, and footer blocks.
- Reads *ELASTIC data as (E, nu, T) and *PLASTIC data as (sigma, eps_p, T).
- Generates an interpolated/extrapolated dataset for a user-defined T_TARGET.
- Plots E(T) and plastic curves.
- Exports new .inp material cards with T_TARGET inserted at the correct position (sorted by temperature).
- Exports one file per method, with the method tag included in the filename.

## Methods (M1/M2/M3)
- M1: Local piecewise linear in temperature (interpolation/extrapolation using the two bracketing points or edge segments). Use for conservative, stable behavior when data are sparse.
- M2: Local quadratic in temperature (fit using the 3 nearest temperature points). Use only near the existing range; it can be unstable far outside the data.
- M3: Scale-by-yield (uses a reference curve shape and scales stress by the sigma_y(T) ratio). Assumes the plastic curve shape is roughly preserved with temperature; use when you want to retain a known curve shape and only scale stresses.

## Key behaviors / Important notes
- Abaqus input expects temperatures sorted; the exported cards are sorted and T_TARGET is inserted in order.
- If *PLASTIC epsilon grids differ by temperature, the tool builds a common target epsilon domain using EPS_DOMAIN_MODE and interpolates each curve onto it.
- Sigma is forced to be non-decreasing with epsilon to avoid non-physical softening in the exported curves.
- Results depend on input quality and are intended for sensitivity studies, not for validated design allowables.

## Requirements
- Python 3.10+
- numpy
- matplotlib

## Quick start
1) Paste your Abaqus material card text into MATERIAL_TEXT.
2) Set T_TARGET.
3) Toggle EXPORT_FIGURES and EXPORT_CARDS (and SHOW_FIGURES if desired).
4) Run the script.

```bash
python material_temp_extrapolator.py
```

## Configuration parameters
- MATERIAL_TEXT: Raw Abaqus material card text containing *ELASTIC and *PLASTIC blocks.
- T_TARGET: Target temperature for interpolation/extrapolation (degC).
- EXPORT_FIGURES: If True, save plots to PNG.
- SHOW_FIGURES: If True, display plots interactively.
- EXPORT_CARDS: If True, write new .inp material cards.
- OUTPUT_PREFIX: Prefix applied to figure and card filenames.
- SCALE_REF_T: Reference temperature for M3 ("closest" or a specific temperature value).
- EPS_MATCH_TOL: Tolerance for determining if *PLASTIC epsilon grids match.
- EPS_DOMAIN_MODE: "intersection" to use common overlap, or "reference" to follow the closest curve shape.
- EPS_EXTRAP_MODE: Behavior outside epsilon range: "nan", "clamp", or "linear".
- EPS_INTERSECTION_NPTS: Number of points used for the intersection epsilon grid.

## Example (generic)
```text
*ELASTIC, TYPE=ISOTROPIC
  210000., 0.30, 20.
  205000., 0.30, 50.
*PLASTIC
  250., 0.0, 20.
  300., 0.02, 20.
  230., 0.0, 50.
  280., 0.02, 50.
  ...
```

## Output artifacts
- Figures are saved in the working directory using OUTPUT_PREFIX, for example:
  - OUT_Elastic_E_vs_T_T<temp>.png
  - OUT_Plastic_curves_T<temp>.png
- Exported cards are named like:
  - OUT_<MaterialName>_T<temp>_M1_LINEAR_LOCAL.inp
  - OUT_<MaterialName>_T<temp>_M2_QUADRATIC_LOCAL.inp
  - OUT_<MaterialName>_T<temp>_M3_SCALE_BY_YIELD_ref<refT>.inp
- Each exported .inp starts with a header comment that logs the method, T_TARGET, and epsilon handling modes.

## Limitations / Disclaimer
- This tool is not validated for any specific material system.
- It is not a substitute for experimental characterization or certified material data.
- Extrapolation uncertainty grows the further T_TARGET is from the input range.
- You must review the generated curves before use in any analysis.

## License
MIT License. See LICENSE.
