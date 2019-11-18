"""Implement ALE meta-analysis."""

import nimare
from nimare.dataset import Dataset


def fit(ds_dict):
    ds = Dataset(ds_dict)
    ma = nimare.meta.ibma.Fishers()
    res = ma.fit(ds)

    z = res.get_map('z')
    p = res.get_map('p')

    return z, p
