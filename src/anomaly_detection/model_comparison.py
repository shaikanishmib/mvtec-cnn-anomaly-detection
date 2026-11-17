"""Saved-artifact comparison; deliberately contains no model or inference code."""
from itertools import combinations
import numpy as np
import pandas as pd

MODELS = ('basic_cnn', 'autoencoder', 'patchcore', 'supersimplenet', 'padim', 'efficientad')
NAMES = dict(zip(MODELS, ('Basic CNN', 'Convolutional AE', 'Simplified PatchCore',
                         'SuperSimpleNet Lite', 'PaDiM', 'EfficientAD-S reduced-budget')))
METRICS = ('image_auroc', 'image_ap', 'pixel_auroc', 'pixel_ap', 'au_pro',
           'recall', 'fpr', 'precision', 'f1')
SEEDS = (2026, 2027, 2028)


def validate_coverage(frame):
    from .data import CATEGORIES
    expected = {(m, c, s) for m in MODELS for c in CATEGORIES for s in SEEDS}
    observed = set(frame[['model', 'category', 'seed']].itertuples(index=False, name=None))
    if len(frame) != 270 or observed != expected or frame.duplicated(['model', 'category', 'seed']).any():
        raise ValueError('Expected exactly six models x 15 categories x three seeds')
    values = frame[list(METRICS)].to_numpy(float)
    # Exact AP integration may exceed one by a floating-point rounding ulp.
    # Accept that numerical tolerance without clipping historical metric values.
    if not np.isfinite(values).all() or ((values < -1e-12) | (values > 1+1e-12)).any():
        raise ValueError('Invalid recorded metric')
    return True


def category_macro(frame):
    """Equal category weighting, averaging seeds within each category first."""
    return frame.groupby(['model', 'category'])[list(METRICS)].mean().groupby('model').mean().reindex(MODELS)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def paired_statistics(frame, draws=10000, seed=9009):
    """Resample whole categories; seeds are averaged, never independent test sets.

    Sign-randomization tests require exchangeable paired category differences.
    The bootstrap describes category sampling, not a new industrial population.
    """
    means = frame.groupby(['model', 'category'])[list(METRICS)].mean()
    categories = sorted(frame.category.unique())
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(categories), (draws, len(categories)))
    signs = 2 * ((np.arange(2**len(categories))[:, None] >> np.arange(len(categories))) & 1) - 1
    rows = []
    for a, b in combinations(MODELS, 2):
        for metric in METRICS:
            d = (means.loc[a].loc[categories, metric] - means.loc[b].loc[categories, metric]).to_numpy()
            effect = d if metric != 'fpr' else -d
            boot = d[samples].mean(axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            p = np.mean(np.abs(signs @ d / len(d)) >= abs(d.mean()) - 1e-14)
            rows.append(dict(model_a=a, model_b=b, metric=metric, difference=d.mean(),
                             ci_low=lo, ci_high=hi, median_difference=np.median(d),
                             paired_dz=d.mean()/d.std(ddof=1) if d.std(ddof=1) else np.nan,
                             favorable_categories=int((effect > 1e-12).sum()),
                             tied_categories=int((np.abs(effect) <= 1e-12).sum()),
                             unfavorable_categories=int((effect < -1e-12).sum()),
                             p_sign_randomization=p, categories=len(d), bootstrap_draws=draws))
    result = pd.DataFrame(rows)
    result['p_holm_135'] = holm(result.p_sign_randomization)
    return result


def threshold_diagnostics(scores, threshold):
    """Describe an already frozen cutoff; never return a newly fitted policy."""
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or not len(s) or not np.isfinite(s).all() or not np.isfinite(threshold):
        raise ValueError('Finite calibration data required')
    return dict(calibration_n=len(s), calibration_fpr=float(np.mean(s > threshold)),
                calibration_exceedances=int((s > threshold).sum()),
                calibration_unique=int(len(np.unique(s))), threshold_ties=int((s == threshold).sum()),
                calibration_min=float(s.min()), calibration_median=float(np.median(s)),
                calibration_max=float(s.max()), calibration_std=float(s.std()))


def normalize_units(seconds, byte_count):
    return dict(hours=float(seconds)/3600, mib=float(byte_count)/2**20, gib=float(byte_count)/2**30)
