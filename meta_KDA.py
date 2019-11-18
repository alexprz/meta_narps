"""Implement ALE meta-analysis."""

import nimare
from nimare.dataset import Dataset
import numpy as np


def fit(ds_dict):
    ds = Dataset(ds_dict)
    ma = nimare.meta.cbma.mkda.KDA()
    res = ma.fit(ds)

    of = res.get_map('of')

    p = _get_p_map(ma, res)
    p = res.masker.inverse_transform(p)

    return of, p


def _get_p_map(estimator, result):
    cor_imgs = estimator._fwe_correct_permutation(result, n_iters=20)
    log_p = cor_imgs['logp_level-voxel']
    return np.exp(-log_p)
