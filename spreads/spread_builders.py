#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build and save ForMoSA posterior-spread ``.npz`` files.

Public entry points: ``physical_posterior_spread()``, ``covariance_noise_spread()``.

Overview
--------
Two complementary spread types are provided, both operating on the same
ForMoSA results folder and producing ``.npz`` files with the same key layout.

``physical_posterior_spread()`` — *physical posterior spread*
    Draws :math:`N` parameter vectors :math:`\theta` from the nested-sampling
    posterior, re-evaluates the atmospheric model at each draw via ForMoSA's
    ``build_models_from_theta``, and computes 1/2/3-sigma envelopes over the
    resulting spectral chain.  Uses ``ProcessPoolExecutor`` for parallelism —
    not suited for use inside notebooks.

``covariance_noise_spread()`` — *observational noise spread*
    Fixes the best-fit model and draws synthetic observations from the
    observational noise law (diagonal or full covariance), then computes
    envelopes over those noise realisations.  Fully serial with no import-time
    side effects — safe to use inside notebooks.

Typical notebook usage
----------------------
::

    from pathlib import Path
    from spread_builders import covariance_noise_spread

    covariance_noise_spread(
        results_folder=Path("/path/to/formosa/results"),
        noise_draws=300,
        # config_path defaults to results_folder / "new_config.ini"
        # output_path defaults to results_folder / "covariance_noise_spread.npz"
    )

Script usage
------------
::

    python spread_builders.py   # set MODE inside __main__ to choose which to run

Output ``.npz`` structure
--------------------------
Both functions write a single ``all_mods`` array of object-dtype dicts, one
dict per observation (or one combined dict when ``combination=True``).  Each
dict contains at minimum:

``wav``, ``best_fit``
    Model wavelength grid and best-fit spectrum (float32).
``spread_{1,2,3}_{low,high}``
    1/2/3-sigma envelope bounds (float32).
``obs_wave``, ``obs_flux``
    Observed wavelength grid and flux saved alongside the model for
    self-contained downstream use (float32).
"""

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ForMoSA.analysis import Analysis
from ForMoSA.config.global_config import ConfigLoader

# Shared state for multiprocessing workers (see _init_worker).
_WORKER_ANALYSIS = None

# ============================================================
# HELPERS
# ============================================================

# -- physical_posterior_spread helpers --

def _get_posterior_samples_and_weights(analysis):
    """Return posterior samples, normalised weights, and parameter names.

    Strips burn-in samples, removes any rows with non-finite weights or fluxes,
    and renormalises the surviving weights to sum to 1.

    Parameters
    ----------
    analysis : ForMoSA.analysis.Analysis
        A fitted ForMoSA analysis object (``adapted=True``, ``fitted=True``).

    Returns
    -------
    samples : np.ndarray, shape (N, D)
        Valid posterior sample matrix.
    weights : np.ndarray, shape (N,)
        Normalised importance weights corresponding to each row of ``samples``.
    parameter_names : list[str]
        Names of the free parameters (columns of ``samples``).
    """

    results = analysis.ns.results

    samples = results.samples[results.burn_in:]
    weights = results.weights[results.burn_in:]
    parameter_names = results.free_parameters

    valid = np.isfinite(weights) & (weights > 0)
    valid &= np.all(np.isfinite(samples), axis=1)
    samples = samples[valid]
    weights = weights[valid]

    weights = weights / weights.sum()

    return samples, weights, parameter_names


def _init_worker(config_path, log_level):
    """Initialise a worker process by loading the ForMoSA analysis into a global.

    Called once per worker by ``ProcessPoolExecutor`` before any tasks are
    dispatched.  Stores the loaded ``Analysis`` object in the module-level
    ``_WORKER_ANALYSIS`` global so that ``_compute_models_from_theta_worker``
    can access it without pickling it on every call.

    Parameters
    ----------
    config_path : Path or str
        Path to the ForMoSA ``.ini`` configuration file.
    log_level : str
        Logging verbosity passed to ``ConfigLoader`` and ``Analysis``
        (e.g. ``"warning"``).
    """

    global _WORKER_ANALYSIS

    config = ConfigLoader(str(config_path), log_level=log_level).load()

    _WORKER_ANALYSIS = Analysis(
        config["config_path"],
        adapted=True,
        fitted=True,
        log_level=log_level,
    )


def _compute_models_from_theta_worker(args):
    """Evaluate the atmospheric forward model for one posterior parameter vector.

    Intended to be dispatched by ``ProcessPoolExecutor.map``.  Uses the
    ``_WORKER_ANALYSIS`` global loaded by ``_init_worker``.

    Parameters
    ----------
    args : np.ndarray, shape (D,)
        A single posterior parameter vector :math:`\theta`.

    Returns
    -------
    packed_models : list[tuple[np.ndarray, np.ndarray]]
        One ``(wave, total_flux)`` pair per observation, where ``total_flux``
        is cast to float32 to reduce IPC overhead.
    """

    theta = args

    models = _WORKER_ANALYSIS.ns_analysis.build_models_from_theta(
        np.asarray(theta, dtype=float)
    )

    packed_models = [
        (m.wave, m.total_flux.astype(np.float32))
        for m in models
    ]
    return packed_models

# -- covariance_noise_spread helpers --

def _build_observational_noise_draws(
    rng,
    obs_name,
    loglike,
    flx_mod,
    obs_flux,
    err,
    cov,
    inv_cov,
    n_noise_draws,
):
    """Draw synthetic observations around a fixed best-fit model.

    Dispatches to diagonal or full-covariance noise sampling depending on the
    ``loglike`` string and whether a covariance matrix is present.  When
    ``noisescaling`` appears in ``loglike``, the noise amplitude is inflated by
    the empirical chi-squared-based scaling factor :math:`s^2` before
    drawing.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded NumPy random generator.
    obs_name : str
        Observation name, used only in error messages.
    loglike : str
        Lower-cased loglikelihood identifier from ForMoSA
        (e.g. ``"chi2"``, ``"covariance"``, ``"noisescaling"``).
    flx_mod : np.ndarray
        Best-fit model flux evaluated at the observation wavelengths.
    obs_flux : np.ndarray
        Observed flux.
    err : np.ndarray or None
        Diagonal per-pixel uncertainties.  Required when ``covariance`` is not
        in ``loglike``.
    cov : np.ndarray or None
        Full covariance matrix of shape ``(N, N)``.
    inv_cov : np.ndarray or None
        Pre-computed inverse of ``cov``.  Required for noise scaling in
        covariance mode.
    n_noise_draws : int
        Number of synthetic realisations to generate.

    Returns
    -------
    draws : np.ndarray, shape (n_noise_draws, N), float32
        Synthetic observed spectra.
    noise_model : str
        ``"diagonal"`` or ``"covariance"``.
    noise_scaling_factor : float
        The :math:`s^2` factor applied to the noise amplitude
        (1.0 when ``noisescaling`` is not active).

    Raises
    ------
    ValueError
        If the required error/covariance arrays are absent, misshapen, or
        contain no valid pixels.
    """

    residuals = obs_flux - flx_mod

    uses_covariance = ("covariance" in loglike) and (cov is not None)
    uses_noisescaling = "noisescaling" in loglike

    if not uses_covariance:
        if err is None:
            raise ValueError(
                f"Observation {obs_name} has no usable diagonal errors.")

        err = np.asarray(err, dtype=np.float64)
        flx_mod = np.asarray(flx_mod, dtype=np.float64)
        obs_flux = np.asarray(obs_flux, dtype=np.float64)

        if err.shape != flx_mod.shape:
            raise ValueError(
                f"Error shape mismatch for {obs_name}: "
                f"err.shape={err.shape}, flx_mod.shape={flx_mod.shape}"
            )

        valid = (
            np.isfinite(flx_mod)
            & np.isfinite(obs_flux)
            & np.isfinite(err)
            & (err > 0)
        )

        if not np.any(valid):
            raise ValueError(
                f"Observation {obs_name} has no valid diagonal errors.")

        draw_err = err.copy()
        s2 = 1.0

        if uses_noisescaling:
            chi2 = float(np.sum((residuals[valid] / err[valid]) ** 2))
            s2 = chi2 / np.count_nonzero(valid)
            draw_err *= np.sqrt(s2)

        print("  noise model: diagonal")
        print(f"  noise scaling used: {uses_noisescaling}")
        print(f"  noise scaling factor s2: {s2:.6g}")

        draws = np.full((n_noise_draws, flx_mod.size),
                        np.nan, dtype=np.float64)

        draws[:, valid] = (
            flx_mod[None, valid]
            + rng.normal(size=(n_noise_draws, np.count_nonzero(valid)))
            * draw_err[None, valid]
        )

        return draws.astype(np.float32), "diagonal", float(s2)

    cov = np.asarray(cov, dtype=np.float64)
    flx_mod = np.asarray(flx_mod, dtype=np.float64)
    obs_flux = np.asarray(obs_flux, dtype=np.float64)

    if cov.shape != (flx_mod.size, flx_mod.size):
        raise ValueError(
            f"Covariance shape mismatch for {obs_name}: "
            f"cov.shape={cov.shape}, expected={(flx_mod.size, flx_mod.size)}"
        )

    draw_cov = cov.copy()
    s2 = 1.0

    if uses_noisescaling:
        if inv_cov is None:
            raise ValueError(
                f"Observation {obs_name} uses covariance noise scaling, "
                "but inv_cov is missing."
            )

        inv_cov = np.asarray(inv_cov, dtype=np.float64)

        if inv_cov.shape != (flx_mod.size, flx_mod.size):
            raise ValueError(
                f"Inverse covariance shape mismatch for {obs_name}: "
                f"inv_cov.shape={inv_cov.shape}, expected={(flx_mod.size, flx_mod.size)}"
            )

        chi2 = float(residuals @ inv_cov @ residuals)
        s2 = chi2 / len(residuals)
        draw_cov *= s2

    print("  noise model: full covariance")
    print(f"  noise scaling used: {uses_noisescaling}")
    print(f"  noise scaling factor s2: {s2:.6g}")

    draws = rng.multivariate_normal(
        mean=flx_mod,
        cov=draw_cov,
        size=n_noise_draws,
    ).astype(np.float32)

    return draws, "covariance", float(s2)


def _check_same_wavelength_grid(waves, name):
    """Assert that all wavelength grids in *waves* are identical.

    Used when ForMoSA analyses multiple observations simultaneously and their
    grids must be combined into a single envelope.

    Parameters
    ----------
    waves : list[np.ndarray]
        Wavelength arrays to compare.
    name : str
        Label used in error messages to identify the array group.

    Returns
    -------
    np.ndarray
        The common wavelength grid (``waves[0]``).

    Raises
    ------
    ValueError
        If any grid differs in shape or values from the first.
    """
    reference_wave = waves[0]

    for i, wave in enumerate(waves[1:], start=1):
        if wave.shape != reference_wave.shape or not np.allclose(
                wave, reference_wave):
            raise ValueError(
                f"Cannot combine observations: wavelength grid mismatch in {name}[{i}]. "
                f"reference shape={reference_wave.shape}, current shape={wave.shape}"
            )

    return reference_wave


def _combine_diagonal_observations(fluxes, models, errors):
    """Combine multiple same-grid observations by inverse-variance weighting.

    Used when ForMoSA analyses multiple observations simultaneously and
    ``combination=True`` is requested with diagonal-error loglikelihoods.
    Pixels with non-finite flux, model, or error values are excluded from the
    weighted average.

    Parameters
    ----------
    fluxes : array-like, shape (M, N)
        Observed fluxes for *M* observations on a common *N*-pixel grid.
    models : array-like, shape (M, N)
        Best-fit model fluxes for each observation.
    errors : array-like, shape (M, N)
        Per-pixel 1-sigma uncertainties for each observation.

    Returns
    -------
    combined_flux : np.ndarray, shape (N,), float32
    combined_model : np.ndarray, shape (N,), float32
    combined_err : np.ndarray, shape (N,), float32
        Inverse-variance weighted combination and its propagated uncertainty.

    Raises
    ------
    ValueError
        If arrays are misshapen, or if the combined result is fully NaN.
    """
    fluxes = np.asarray(fluxes, dtype=np.float64)
    models = np.asarray(models, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)

    if fluxes.ndim != 2:
        raise ValueError(f"fluxes must be 2D, got shape {fluxes.shape}")

    if models.shape != fluxes.shape:
        raise ValueError(
            f"Model shape mismatch: models.shape={models.shape}, "
            f"fluxes.shape={fluxes.shape}"
        )

    if errors.shape != fluxes.shape:
        raise ValueError(
            f"Error shape mismatch: errors.shape={errors.shape}, "
            f"fluxes.shape={fluxes.shape}"
        )

    valid = (
        np.isfinite(fluxes)
        & np.isfinite(models)
        & np.isfinite(errors)
        & (errors > 0)
    )

    if not np.any(valid):
        raise ValueError("No valid points available for diagonal combination.")

    weights = np.zeros_like(errors, dtype=np.float64)
    weights[valid] = 1.0 / np.square(errors[valid])

    weight_sum = np.sum(weights, axis=0)

    combined_flux = np.full(fluxes.shape[1], np.nan, dtype=np.float64)
    combined_model = np.full(models.shape[1], np.nan, dtype=np.float64)
    combined_err = np.full(errors.shape[1], np.nan, dtype=np.float64)

    good = np.isfinite(weight_sum) & (weight_sum > 0)

    combined_flux[good] = (
        np.sum(weights[:, good] * fluxes[:, good], axis=0) / weight_sum[good]
    )

    combined_model[good] = (
        np.sum(weights[:, good] * models[:, good], axis=0) / weight_sum[good]
    )

    combined_err[good] = 1.0 / np.sqrt(weight_sum[good])

    if not np.any(np.isfinite(combined_flux)):
        raise ValueError("Combined flux is fully NaN.")

    if not np.any(np.isfinite(combined_model)):
        raise ValueError("Combined model is fully NaN.")

    if not np.any(np.isfinite(combined_err)):
        raise ValueError("Combined error is fully NaN.")

    return (
        combined_flux.astype(np.float32),
        combined_model.astype(np.float32),
        combined_err.astype(np.float32),
    )


def _combine_covariance_observations(fluxes, models, covs, inv_covs):
    """Combine multiple same-grid observations by inverse-covariance weighting.

    Used when ForMoSA analyses multiple observations simultaneously and
    ``combination=True`` is requested with covariance-based loglikelihoods.
    The combined covariance is the inverse of the sum of individual inverse
    covariances.

    Parameters
    ----------
    fluxes : list[array-like], each shape (N,)
        Observed fluxes for each observation.
    models : list[array-like], each shape (N,)
        Best-fit model fluxes for each observation.
    covs : list[array-like or None], each shape (N, N)
        Full covariance matrices.  ``None`` entries are skipped if
        ``inv_covs`` provides a direct inverse.
    inv_covs : list[array-like or None], each shape (N, N)
        Pre-computed inverse covariance matrices.  If ``None`` for an entry,
        ``covs[i]`` is inverted numerically.

    Returns
    -------
    combined_flux : np.ndarray, shape (N,), float32
    combined_model : np.ndarray, shape (N,), float32
    combined_cov : np.ndarray, shape (N, N), float32
    combined_inv_cov : np.ndarray, shape (N, N), float32
    combined_err : np.ndarray, shape (N,), float32
        Square root of the diagonal of ``combined_cov``.

    Raises
    ------
    ValueError
        If any observation has neither ``cov`` nor ``inv_cov``.
    """
    fluxes = [np.asarray(flux, dtype=np.float64) for flux in fluxes]
    models = [np.asarray(model, dtype=np.float64) for model in models]

    final_inv_covs = []

    for i, (cov, inv_cov) in enumerate(zip(covs, inv_covs, strict=True)):
        if inv_cov is not None:
            final_inv_cov = np.asarray(inv_cov, dtype=np.float64)
        elif cov is not None:
            final_inv_cov = np.linalg.inv(np.asarray(cov, dtype=np.float64))
        else:
            raise ValueError(
                f"Cannot combine covariance observations: observation {i} has no covariance "
                "or inverse covariance."
            )

        final_inv_covs.append(final_inv_cov)

    combined_inv_cov = np.sum(final_inv_covs, axis=0)
    combined_cov = np.linalg.inv(combined_inv_cov)

    flux_rhs = np.zeros_like(fluxes[0], dtype=np.float64)
    model_rhs = np.zeros_like(models[0], dtype=np.float64)

    for flux, model, inv_cov in zip(
            fluxes, models, final_inv_covs, strict=True):
        flux_rhs += inv_cov @ flux
        model_rhs += inv_cov @ model

    combined_flux = combined_cov @ flux_rhs
    combined_model = combined_cov @ model_rhs
    combined_err = np.sqrt(np.diag(combined_cov))

    return (
        combined_flux.astype(np.float32),
        combined_model.astype(np.float32),
        combined_cov.astype(np.float32),
        combined_inv_cov.astype(np.float32),
        combined_err.astype(np.float32),
    )


def _build_single_model_dict(
    rng,
    obs_name,
    loglike,
    wav,
    flx_mod,
    obs_flux,
    err=None,
    cov=None,
    inv_cov=None,
    obs_wave_all=None,
    obs_flux_all=None,
    obs_err_all=None,
    original_observations=None,
    n_noise_draws=300,
    save_noise_draws=False,
):
    """Build the serialisable result dictionary for one noise-spread spectrum.

    Generates noise draws via :func:`_build_observational_noise_draws`, computes
    1/2/3-sigma percentile envelopes, and assembles all arrays into a flat dict
    suitable for storage with ``np.savez_compressed``.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded NumPy random generator.
    obs_name : str
        Observation identifier stored in the output dict.
    loglike : str
        Lower-cased loglikelihood identifier.
    wav : np.ndarray
        Model (and observation) wavelength grid.
    flx_mod : np.ndarray
        Best-fit model flux.
    obs_flux : np.ndarray
        Observed flux on the same grid as ``wav``.
    err : np.ndarray or None
        Diagonal per-pixel uncertainties.
    cov : np.ndarray or None
        Full covariance matrix.
    inv_cov : np.ndarray or None
        Inverse of ``cov``.
    obs_wave_all : np.ndarray or None
        Original per-observation wavelength grids before combination.
        Written to the dict only when provided (combination mode).
    obs_flux_all : np.ndarray or None
        Original per-observation fluxes before combination.
    obs_err_all : np.ndarray or None
        Original per-observation errors before combination.
    original_observations : list[str] or None
        Names of the individual observations that were combined.
    n_noise_draws : int
        Number of synthetic noise realisations.
    save_noise_draws : bool
        If True, include the raw draw matrix (``noise_draws`` key) in the
        output dict.

    Returns
    -------
    dict
        Keys always present: ``observation``, ``logl_type``, ``noise_model``,
        ``noise_scaling_factor``, ``wav``, ``best_fit``,
        ``spread_{1,2,3}_{low,high}``, ``obs_wave``, ``obs_flux``.
        Optional keys: ``obs_err``, ``obs_wave_all``, ``obs_flux_all``,
        ``obs_err_all``, ``original_observations``, ``noise_draws``.
    """

    draws, noise_model, noise_scaling_factor = _build_observational_noise_draws(
        rng=rng,
        obs_name=obs_name,
        loglike=loglike,
        flx_mod=flx_mod,
        obs_flux=obs_flux,
        err=err,
        cov=cov,
        inv_cov=inv_cov,
        n_noise_draws=n_noise_draws,
    )

    if draws.ndim != 2:
        raise ValueError(
            f"Noise draws for {obs_name} must be 2D, got shape {draws.shape}."
        )

    if draws.shape[1:] != flx_mod.shape:
        raise ValueError(
            f"Noise draw shape mismatch for {obs_name}: "
            f"draws spectral shape={draws.shape[1:]}, flx_mod.shape={flx_mod.shape}"
        )

    print(f"  draws.shape: {draws.shape}")

    pcts = np.nanpercentile(
        draws, [0.135, 2.275, 15.865, 84.135, 97.725, 99.865], axis=0)
    spread_3_low, spread_2_low, spread_1_low, spread_1_high, spread_2_high, spread_3_high = pcts

    width_1s = spread_1_high - spread_1_low
    width_2s = spread_2_high - spread_2_low
    width_3s = spread_3_high - spread_3_low
    scale = np.nanmedian(np.abs(flx_mod))

    print(f"  median |best-fit flux|: {scale:.6e}")
    print(f"  median 1σ noise width: {np.nanmedian(width_1s):.6e}")
    print(f"  median 2σ noise width: {np.nanmedian(width_2s):.6e}")
    print(f"  median 3σ noise width: {np.nanmedian(width_3s):.6e}")

    if np.isfinite(scale) and scale > 0:
        print(
            f"  relative median 1σ width: {np.nanmedian(width_1s) / scale:.6e}")
        print(
            f"  relative median 2σ width: {np.nanmedian(width_2s) / scale:.6e}")
        print(
            f"  relative median 3σ width: {np.nanmedian(width_3s) / scale:.6e}")

    model_dict = {
        "observation": obs_name,
        "logl_type": loglike,
        "noise_model": noise_model,
        "noise_scaling_factor": np.float32(noise_scaling_factor),
        "wav": np.asarray(wav, dtype=np.float32),
        "best_fit": np.asarray(flx_mod, dtype=np.float32),
        "spread_1_low": spread_1_low.astype(np.float32),
        "spread_1_high": spread_1_high.astype(np.float32),
        "spread_2_low": spread_2_low.astype(np.float32),
        "spread_2_high": spread_2_high.astype(np.float32),
        "spread_3_low": spread_3_low.astype(np.float32),
        "spread_3_high": spread_3_high.astype(np.float32),

        # Combined observation, or the single observation in non-combination
        # mode.
        "obs_wave": np.asarray(wav, dtype=np.float32),
        "obs_flux": np.asarray(obs_flux, dtype=np.float32),
    }

    if err is not None:
        model_dict["obs_err"] = np.asarray(err, dtype=np.float32)

    # Original observations saved only in combination mode.
    if obs_wave_all is not None:
        model_dict["obs_wave_all"] = np.asarray(obs_wave_all, dtype=np.float32)

    if obs_flux_all is not None:
        model_dict["obs_flux_all"] = np.asarray(obs_flux_all, dtype=np.float32)

    if obs_err_all is not None:
        model_dict["obs_err_all"] = np.asarray(obs_err_all, dtype=np.float32)

    if original_observations is not None:
        model_dict["original_observations"] = np.asarray(
            original_observations, dtype=object)

    if save_noise_draws:
        model_dict["noise_draws"] = draws.astype(np.float32)

    return model_dict


def _build_individual_model_dicts(
    rng,
    observations,
    best_fit,
    logl_types,
    n_noise_draws=300,
    save_noise_draws=False,
):
    """Build one noise-spread result dictionary per observation.

    Iterates over ``observations`` and ``best_fit`` in lock-step, validating
    wavelength and flux shape consistency before calling
    :func:`_build_single_model_dict` for each.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded NumPy random generator.
    observations : list
        ForMoSA observation objects (provide ``.name``, ``.wave``, ``.flux``,
        ``.err``, ``.cov``, ``.inv_cov``).
    best_fit : list
        ForMoSA best-fit model objects (provide ``.wave``, ``.total_flux``),
        one per observation.
    logl_types : list
        ForMoSA loglikelihood type objects (provide ``.loglike``), one per
        observation.
    n_noise_draws : int
        Number of synthetic noise realisations per observation.
    save_noise_draws : bool
        If True, include the raw draw matrix in each output dict.

    Returns
    -------
    list[dict]
        One result dict per observation, in the same order as ``observations``.
    """

    all_mods = []

    for obs, model, logl_type in zip(
            observations, best_fit, logl_types, strict=True):
        obs_name = obs.name
        loglike = str(logl_type.loglike).lower()

        print(f"Processing observation: {obs_name}")
        print(f"  logL type: {loglike}")

        wav = model.wave
        flx_mod = model.total_flux.astype(np.float32)

        obs_wave = obs.wave
        obs_flux = obs.flux.astype(np.float32)

        if obs_wave.shape != wav.shape or not np.allclose(obs_wave, wav):
            raise ValueError(
                f"Observation/model wavelength mismatch for {obs_name}: "
                f"obs.wave.shape={obs_wave.shape}, model.wave.shape={wav.shape}"
            )

        if obs_flux.shape != flx_mod.shape:
            raise ValueError(
                f"Flux shape mismatch for {obs_name}: "
                f"obs.flux.shape={obs_flux.shape}, flx_mod.shape={flx_mod.shape}"
            )

        model_dict = _build_single_model_dict(
            rng=rng,
            obs_name=obs_name,
            loglike=loglike,
            wav=wav,
            flx_mod=flx_mod,
            obs_flux=obs_flux,
            err=obs.err,
            cov=obs.cov,
            inv_cov=obs.inv_cov,
            n_noise_draws=n_noise_draws,
            save_noise_draws=save_noise_draws,
        )

        all_mods.append(model_dict)

    return all_mods


def _build_combined_model_dict(
    rng,
    observations,
    best_fit,
    logl_types,
    n_noise_draws=300,
    save_noise_draws=False,
):
    """Combine all observations into one spectrum, then build its noise-spread dict.

    All observations must share the same wavelength grid and the same
    loglikelihood type.  Depending on whether the loglikelihood is covariance-
    or diagonal-based, uses :func:`_combine_covariance_observations` or
    :func:`_combine_diagonal_observations` before drawing noise.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded NumPy random generator.
    observations : list
        ForMoSA observation objects (same interface as in
        :func:`_build_individual_model_dicts`).
    best_fit : list
        ForMoSA best-fit model objects, one per observation.
    logl_types : list
        ForMoSA loglikelihood type objects, one per observation.
    n_noise_draws : int
        Number of synthetic noise realisations for the combined spectrum.
    save_noise_draws : bool
        If True, include the raw draw matrix in the output dict.

    Returns
    -------
    dict
        A single result dict for the combined spectrum (same key layout as
        :func:`_build_single_model_dict`), plus ``obs_wave_all``,
        ``obs_flux_all``, ``obs_err_all``, and ``original_observations`` for
        traceability.

    Raises
    ------
    ValueError
        If observations have different loglikelihood types, mismatched
        wavelength grids, or missing error/covariance data.
    """

    loglikes = [str(logl_type.loglike).lower() for logl_type in logl_types]

    if len(set(loglikes)) != 1:
        raise ValueError(
            "Cannot combine observations with different logL types: "
            f"{loglikes}"
        )

    loglike = loglikes[0]
    uses_covariance = "covariance" in loglike

    obs_names = [str(obs.name) for obs in observations]
    combined_name = f"combined_{len(observations)}obs"

    print(f"Processing combined observation: {combined_name}")
    print(f"  original observations: {obs_names}")
    print(f"  logL type: {loglike}")
    print(f"  number of combined observations: {len(observations)}")

    model_waves = [model.wave for model in best_fit]
    obs_waves = [obs.wave for obs in observations]

    common_model_wave = _check_same_wavelength_grid(model_waves, "model.wave")
    common_obs_wave = _check_same_wavelength_grid(obs_waves, "obs.wave")

    if common_obs_wave.shape != common_model_wave.shape or not np.allclose(
        common_obs_wave, common_model_wave
    ):
        raise ValueError(
            "Cannot combine observations: observation and model wavelength grids differ."
        )

    fluxes, models, errors, covs, inv_covs = [], [], [], [], []

    for obs, model in zip(observations, best_fit, strict=True):
        obs_flux = obs.flux.astype(np.float64)
        flx_mod = model.total_flux.astype(np.float64)
        if obs_flux.shape != common_obs_wave.shape:
            raise ValueError(
                f"Flux shape mismatch for {obs.name}: {obs_flux.shape} vs {common_obs_wave.shape}")
        if flx_mod.shape != common_obs_wave.shape:
            raise ValueError(
                f"Model shape mismatch for {obs.name}: {flx_mod.shape} vs {common_obs_wave.shape}")
        fluxes.append(obs_flux)
        models.append(flx_mod)
        errors.append(obs.err.astype(np.float64)
                      if obs.err is not None else None)
        covs.append(obs.cov)
        inv_covs.append(obs.inv_cov)

    obs_wave_all = np.asarray(obs_waves, dtype=np.float64)
    obs_flux_all = np.asarray(fluxes, dtype=np.float64)

    if all(err is not None for err in errors):
        obs_err_all = np.asarray(errors, dtype=np.float64)
    else:
        obs_err_all = None

    if uses_covariance:
        combined_flux, combined_model, combined_cov, combined_inv_cov, combined_err = (
            _combine_covariance_observations(
                fluxes=fluxes,
                models=models,
                covs=covs,
                inv_covs=inv_covs,
            )
        )

        print("  combination method: inverse covariance weighted")
        print(f"  combined median error: {np.nanmedian(combined_err):.6e}")

        return _build_single_model_dict(
            rng=rng,
            obs_name=combined_name,
            loglike=loglike,
            wav=common_model_wave,
            flx_mod=combined_model,
            obs_flux=combined_flux,
            err=combined_err,
            cov=combined_cov,
            inv_cov=combined_inv_cov,
            obs_wave_all=obs_wave_all,
            obs_flux_all=obs_flux_all,
            obs_err_all=obs_err_all,
            original_observations=obs_names,
            n_noise_draws=n_noise_draws,
            save_noise_draws=save_noise_draws,
        )

    if any(err is None for err in errors):
        raise ValueError(
            "Cannot combine diagonal observations because at least one observation "
            "has no diagonal error."
        )

    print("  diagonal error diagnostics before combination:")

    for i, err in enumerate(errors):
        err = np.asarray(err, dtype=np.float64)
        print(
            f"    obs {i}: err min/p1/median/max = "
            f"{np.nanmin(err):.6e} / "
            f"{np.nanpercentile(err, 1):.6e} / "
            f"{np.nanmedian(err):.6e} / "
            f"{np.nanmax(err):.6e}"
        )

    combined_flux, combined_model, combined_err = _combine_diagonal_observations(
        fluxes=fluxes,
        models=models,
        errors=errors,
    )

    print("  combination method: inverse variance weighted")
    print(
        f"  combined err min/p1/median/max = "
        f"{np.nanmin(combined_err):.6e} / "
        f"{np.nanpercentile(combined_err, 1):.6e} / "
        f"{np.nanmedian(combined_err):.6e} / "
        f"{np.nanmax(combined_err):.6e}"
    )

    return _build_single_model_dict(
        rng=rng,
        obs_name=combined_name,
        loglike=loglike,
        wav=common_model_wave,
        flx_mod=combined_model,
        obs_flux=combined_flux,
        err=combined_err,
        cov=None,
        inv_cov=None,
        obs_wave_all=obs_wave_all,
        obs_flux_all=obs_flux_all,
        obs_err_all=obs_err_all,
        original_observations=obs_names,
        n_noise_draws=n_noise_draws,
        save_noise_draws=save_noise_draws,
    )

# ============================================================
# MAIN FUNCTIONS
# ============================================================


def physical_posterior_spread(
    config_path,
    output_path,
    n_posterior_models=300,
    n_workers=5,
    random_seed=None,
    save_model_chain=False,
    log_level="warning",
) -> None:
    """Build and save 1/2/3-sigma physical posterior spectral envelopes.

    Draws ``n_posterior_models`` parameter vectors from the nested-sampling
    posterior (weighted by importance weights), evaluates the atmospheric
    forward model at each draw in parallel using ``ProcessPoolExecutor``, and
    computes percentile envelopes over the resulting spectral chain.

    .. warning::
        This function sets ``OMP_NUM_THREADS`` and related environment
        variables to ``"1"`` to prevent BLAS oversubscription when forking
        workers.  It should not be called from inside a Jupyter notebook kernel;
        use :func:`covariance_noise_spread` for notebook workflows.

    Parameters
    ----------
    config_path : str or Path
        Path to the ForMoSA ``.ini`` configuration file for the run to process.
    output_path : str or Path
        Destination path for the output ``.npz`` file.
    n_posterior_models : int, default 300
        Number of posterior parameter vectors to draw and evaluate.
    n_workers : int, default 5
        Number of parallel worker processes.
    random_seed : int or None, default None
        Seed for the NumPy random generator used to draw posterior samples.
        ``None`` gives non-reproducible results.
    save_model_chain : bool, default False
        If True, include the full ``(n_posterior_models, N_pixels)`` spectral
        chain in the output ``.npz`` under the key ``model_chain``.
        Significantly increases file size.
    log_level : str, default ``"warning"``
        Logging verbosity passed to ForMoSA's ``ConfigLoader`` and ``Analysis``.

    Returns
    -------
    None
        Results are written to ``output_path``.  The file contains a single
        ``all_mods`` array of object-dtype dicts, one per observation.

    Raises
    ------
    ValueError
        If the model wavelength grid changes between posterior draws for the
        same observation, or if shapes are inconsistent.

    Examples
    --------
    ::

        from spread_builders import physical_posterior_spread

        physical_posterior_spread(
            config_path="/path/to/config.ini",
            output_path="/path/to/results/physical_posterior_spread.npz",
            n_posterior_models=500,
            n_workers=4,
            random_seed=42,
        )
    """

    # Set thread counts before spawning workers to avoid BLAS oversubscription.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    print(f"Loading analysis from: {config_path}")
    print(f"Using {n_workers} worker processes")

    config = ConfigLoader(str(config_path), log_level=log_level).load()

    analysis = Analysis(
        config["config_path"],
        adapted=True,
        fitted=True,
        log_level=log_level,
    )

    rng = np.random.default_rng(random_seed)

    samples, weights, parameter_names = _get_posterior_samples_and_weights(
        analysis)

    observations = analysis.ns.restricted_observations
    logl_types = analysis.ns.logL_type
    best_fit = analysis.ns_analysis.best_fit

    print(f"Posterior samples available: {samples.shape[0]}")
    print(f"Posterior sample dimension: {samples.shape[1]}")
    print(f"Number of observations: {len(observations)}")

    if parameter_names:
        print(f"Posterior parameter names: {parameter_names}")
    else:
        print("Posterior parameter names: not available")

    selected_indices = rng.choice(
        len(samples),
        size=n_posterior_models,
        replace=True,
        p=weights,
    )
    selected_thetas = samples[selected_indices]

    print(
        f"Posterior models: {n_posterior_models} x {len(observations)} observations = {n_posterior_models * len(observations)} total")

    # Store observation/best-fit info in simple variables.
    # This keeps the rest of the script independent from the parent Analysis
    # object.
    obs_names = [obs.name for obs in observations]
    obs_waves = [obs.wave.astype(np.float32) for obs in observations]
    obs_fluxes = [obs.flux.astype(np.float32) for obs in observations]

    obs_errs = [obs.err for obs in observations]

    best_fit_fluxes = [
        model.total_flux.astype(
            np.float32) for model in best_fit]

    loglike_names = [str(logl_type.loglike) for logl_type in logl_types]

    model_chains = [[] for _ in observations]
    wav_by_obs = [None] * len(observations)

    tasks = list(selected_thetas)

    print("Generating posterior models in parallel...")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(config_path, log_level),
    ) as executor:
        for done_count, packed_models in enumerate(
            executor.map(_compute_models_from_theta_worker, tasks, chunksize=1),
            start=1,
        ):
            for obs_index, (model_wav, model_flux) in enumerate(packed_models):
                obs_name = obs_names[obs_index]

                if wav_by_obs[obs_index] is None:
                    wav_by_obs[obs_index] = model_wav
                elif (
                    model_wav.shape != wav_by_obs[obs_index].shape
                    or not np.allclose(model_wav, wav_by_obs[obs_index])
                ):
                    raise ValueError(
                        f"Model wavelength grid changed for {obs_name}: "
                        f"first shape={wav_by_obs[obs_index].shape}, "
                        f"current shape={model_wav.shape}"
                    )

                model_chains[obs_index].append(model_flux)

            print(
                f"  Done {done_count}/{len(selected_thetas)} posterior models",
                flush=True,
            )

    all_mods = []

    for obs_index, obs_name in enumerate(obs_names):
        loglike = loglike_names[obs_index].lower()

        print(f"Processing observation: {obs_name}")
        print(f"  logL type: {loglike}")

        wav = wav_by_obs[obs_index]

        flx_model_chain = np.asarray(model_chains[obs_index], dtype=np.float32)

        obs_flux = obs_fluxes[obs_index]

        if obs_flux.shape != flx_model_chain.shape[1:]:
            raise ValueError(
                f"Shape mismatch for {obs_name}: "
                f"obs.flux.shape={obs_flux.shape}, "
                f"model_chain spectral shape={flx_model_chain.shape[1:]}"
            )

        print(f"  flx_model_chain.shape: {flx_model_chain.shape}")

        best_fit_flux = best_fit_fluxes[obs_index]

        median_spec = np.percentile(flx_model_chain, 50, axis=0)
        spec_1_low, spec_1_high = np.percentile(
            flx_model_chain, [15.865, 84.135], axis=0)
        spec_2_low, spec_2_high = np.percentile(
            flx_model_chain, [2.275, 97.725], axis=0)
        spec_3_low, spec_3_high = np.percentile(
            flx_model_chain, [0.135, 99.865], axis=0)

        model_dict = {
            "observation": obs_name,
            "logl_type": loglike_names[obs_index],

            # Model wavelength grid and posterior envelopes
            "wav": wav.astype(np.float32),
            "median": median_spec.astype(np.float32),
            "best_fit": best_fit_flux.astype(np.float32),
            "spread_1_low": spec_1_low.astype(np.float32),
            "spread_1_high": spec_1_high.astype(np.float32),
            "spread_2_low": spec_2_low.astype(np.float32),
            "spread_2_high": spec_2_high.astype(np.float32),
            "spread_3_low": spec_3_low.astype(np.float32),
            "spread_3_high": spec_3_high.astype(np.float32),

            # Observation saved too, so the .npz is self-contained
            "obs_wave": obs_waves[obs_index],
            "obs_flux": obs_fluxes[obs_index],
        }

        model_dict["obs_err"] = obs_errs[obs_index]

        if save_model_chain:
            model_dict["model_chain"] = flx_model_chain.astype(np.float32)

        all_mods.append(model_dict)

    print(
        f"Built physical posterior envelopes for {len(all_mods)} observations.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        all_mods=np.asarray(
            all_mods,
            dtype=object))
    print(f"Saved spread data to: {output_path}")

# ============================================================
# PUBLIC API
# ============================================================


def covariance_noise_spread(
    results_folder,
    config_path=None,
    output_path=None,
    noise_draws=300,
    random_seed=None,
    save_noise_draws=False,
    combination=False,
    log_level="warning",
):
    """Build and save 1/2/3-sigma observational-noise spectral envelopes.

    Fixes the best-fit model from a ForMoSA results folder and generates
    ``noise_draws`` synthetic observations by sampling from the observational
    noise law (diagonal or full covariance).  Percentile envelopes over the
    draws are then saved to a ``.npz`` file.

    This function is serial and has no import-time side effects; it is safe
    to call from inside a Jupyter notebook.

    Parameters
    ----------
    results_folder : str or Path
        Path to the ForMoSA results folder.  Used to resolve default paths
        for ``config_path`` and ``output_path``.
    config_path : str or Path or None, default None
        Path to the ForMoSA ``.ini`` file.  Defaults to
        ``results_folder / "new_config.ini"``.
    output_path : str or Path or None, default None
        Destination ``.npz`` path.  Defaults to
        ``results_folder / "covariance_noise_spread.npz"``.
        When ``combination=True``, ``_combined`` is appended to the stem.
    noise_draws : int, default 300
        Number of synthetic noise realisations to draw per spectrum.
    random_seed : int or None, default None
        Seed for the NumPy random generator.  ``None`` gives non-reproducible
        results.
    save_noise_draws : bool, default False
        If True, include the full ``(noise_draws, N_pixels)`` draw matrix in
        the output under the key ``noise_draws``.  Significantly increases
        file size.
    combination : bool, default False
        If True, combine all observations into a single weighted-average
        spectrum before drawing noise (intended for multi-observation
        inversions).  The combined result is stored as a single dict.
    log_level : str, default ``"warning"``
        Logging verbosity passed to ForMoSA's ``ConfigLoader`` and ``Analysis``.

    Returns
    -------
    Path
        Absolute path to the written ``.npz`` file.

    Raises
    ------
    ValueError
        If ``noise_draws <= 0``, or if any observation has no usable
        error/covariance data.

    Examples
    --------
    ::

        from pathlib import Path
        from spread_builders import covariance_noise_spread

        out = covariance_noise_spread(
            results_folder=Path("/path/to/results"),
            noise_draws=300,
            random_seed=0,
        )
        print(f"Saved to: {out}")
    """

    if noise_draws <= 0:
        raise ValueError("noise_draws must be > 0.")

    results_folder = Path(results_folder)

    if config_path is None:
        config_path = results_folder / "new_config.ini"
    else:
        config_path = Path(config_path)

    if output_path is None:
        output_path = results_folder / "covariance_noise_spread.npz"
    else:
        output_path = Path(output_path)

    if combination:
        output_path = output_path.with_name(
            f"{output_path.stem}_combined{output_path.suffix}")

    print(f"Loading analysis from: {config_path}")
    print(f"Combination mode: {combination}")
    print(f"Output path: {output_path}")

    config = ConfigLoader(str(config_path), log_level=log_level).load()

    analysis = Analysis(
        config["config_path"],
        adapted=True,
        fitted=True,
        log_level=log_level,
    )

    rng = np.random.default_rng(random_seed)

    observations = analysis.ns.restricted_observations
    logl_types = analysis.ns.logL_type
    best_fit = analysis.ns_analysis.best_fit

    print(f"Number of observations: {len(observations)}")
    print(f"Observational noise draws requested: {noise_draws}")

    if combination:
        combined_model_dict = _build_combined_model_dict(
            rng=rng,
            observations=observations,
            best_fit=best_fit,
            logl_types=logl_types,
            n_noise_draws=noise_draws,
            save_noise_draws=save_noise_draws,
        )
        all_mods = [combined_model_dict]
    else:
        all_mods = _build_individual_model_dicts(
            rng=rng,
            observations=observations,
            best_fit=best_fit,
            logl_types=logl_types,
            n_noise_draws=noise_draws,
            save_noise_draws=save_noise_draws,
        )

    print(
        f"Built observational-noise envelopes for {len(all_mods)} spectrum/spectra.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        all_mods=np.asarray(
            all_mods,
            dtype=object))

    print(f"Saved covariance/noise spread data to: {output_path}")
    return output_path


if __name__ == "__main__":
    # ── Shared parameters ────────────────────────────────────
    RESULTS_FOLDER = Path(
        "/home/localuser/Stage_ForMoSA_desk/aflepb_extra/results/d50/1.5e-4/_30deg"
    )
    CONFIG_PATH = "/home/localuser/Stage_ForMoSA_desk/aflepb_extra/configs/d50/1.5e-4/config_30deg.ini"
    RANDOM_SEED = None

    # ── Choose which spread to build ──────────────────────────
    MODE = "covariance"   # "physical" or "covariance"

    # ── Physical posterior spread parameters ──────────────────
    N_POSTERIOR_MODELS = 300
    N_WORKERS = 2
    SAVE_MODEL_CHAIN = False
    OUTPUT_PATH_PHYSICAL = RESULTS_FOLDER / "physical_posterior_spread.npz"

    # ── Covariance / noise spread parameters ──────────────────
    N_NOISE_DRAWS = 300
    SAVE_NOISE_DRAWS = False
    # combine all observations into one envelope (4obs cases)
    COMBINATION = False
    OUTPUT_PATH_COVARIANCE = RESULTS_FOLDER / "covariance_noise_spread.npz"

    # ── Dispatch ──────────────────────────────────────────────
    if MODE == "physical":
        physical_posterior_spread(
            config_path=CONFIG_PATH,
            output_path=OUTPUT_PATH_PHYSICAL,
            n_posterior_models=N_POSTERIOR_MODELS,
            n_workers=N_WORKERS,
            random_seed=RANDOM_SEED,
            save_model_chain=SAVE_MODEL_CHAIN,
        )
    elif MODE == "covariance":
        covariance_noise_spread(
            results_folder=RESULTS_FOLDER,
            config_path=CONFIG_PATH,
            output_path=OUTPUT_PATH_COVARIANCE,
            noise_draws=N_NOISE_DRAWS,
            random_seed=RANDOM_SEED,
            save_noise_draws=SAVE_NOISE_DRAWS,
            combination=COMBINATION,
        )
    else:
        raise ValueError(
            f"Unknown MODE {MODE!r}. Use 'physical' or 'covariance'.")