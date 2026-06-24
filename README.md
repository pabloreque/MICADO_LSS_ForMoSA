# MICADO_LSS_ForMoSA

Python analysis and visualisation tools developed during a research internship at the Observatoire de Paris (LIRA), focused on characterising the spectroscopic capabilities of MICADO for the atmospheric retrieval of directly imaged exoplanets.

## Scientific context

[MICADO](https://elt.eso.org/instrument/MICADO/) is the first-light near-infrared instrument of the Extremely Large Telescope (ELT). This project assesses its long-slit spectroscopic (LSS) mode in the H+K band for the atmospheric characterisation of directly imaged exoplanets, using a simulated analogue of AF Lep b as a test case.

The simulation and analysis pipeline combines:
- **MISTHIC** — PSF simulator for MICADO high-contrast modes
- **[Exo-REM](https://gitlab.obspm.fr/Exoplanet-Atmospheres-LESIA/exorem)** — self-consistent radiative-convective atmospheric model grid
- **[ForMoSA](https://github.com/exoAtmospheres/ForMoSA)** — Bayesian nested-sampling atmospheric retrieval tool

## Contents

| File | Description |
|------|-------------|
| `corner_plots.py` | Multi-run posterior comparison corner plots for ForMoSA outputs |

## Dependencies

- Python 3.11
- `numpy`, `matplotlib`, `corner`
- [ForMoSA](https://github.com/exoAtmospheres/ForMoSA) (must be installed and importable)

## Author

Pablo Requeijo — M1 LIU, Master SUTS, Observatoire de Paris PSL  
Internship at LIRA (April–June 2026), supervised by Paulina Palma-Bifani and Pierre Baudoz.
