"""Implement ALE meta-analysis."""

import nimare
from nimare.dataset import Dataset


def fit(ds_dict):
    ds = Dataset(ds_dict)
    ma = nimare.meta.cbma.ale.ALE()
    res = ma.fit(ds)

    img_ale = res.get_map('ale')
    img_p = res.get_map('p')
    # img_z = res.get_map('z')

    return img_ale, img_p


