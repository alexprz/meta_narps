from joblib import Memory
from nilearn import plotting
import matplotlib
from matplotlib import pyplot as plt
import copy
import nibabel as nib
import nimare

from data_utils import retrieve_imgs, iterator_imgs, iterator_paths_imgs, \
    iterator_name_imgs
from extract import get_activations
from meta_formatter import dict_from_activations, dict_from_paths
from process import process
import compare
import img_utils

import meta_ALE
import meta_KDA
import meta_MKDA
import meta_Stouffers
import meta_Fishers


matplotlib.use('MacOSX')

cache_dir = 'cache'
raw_data_dir = 'data/orig/'
proc_data_dir = 'data/proc/'

get_activations = Memory(cache_dir).cache(get_activations)


def fdr_threshold(img_list, img_p, q=0.05):
    """Compute FDR and threshold same-sized images."""
    arr_list = [copy.copy(img.get_fdata()) for img in img_list]
    arr_p = img_p.get_fdata()
    aff = img_p.affine

    fdr = nimare.stats.fdr(arr_p.ravel(), q=q)

    for arr in arr_list:
        arr[arr_p > fdr] = 0

    res_list = [nib.Nifti1Image(arr, aff) for arr in arr_list]

    return res_list


def run_meta(filename, verbose=False):
    raw_paths = retrieve_imgs(raw_data_dir, filename)

    # Process raw data
    iter_name_imgs = iterator_name_imgs(raw_paths)
    process(iter_name_imgs, proc_data_dir, filename, verbose=verbose)

    proc_paths = retrieve_imgs(proc_data_dir, filename)
    iter_imgs = iterator_imgs(proc_paths)

    activations = get_activations(list(iter_imgs), 1.96, verbose=verbose)

    cbma_dict = dict_from_activations(activations, 119)
    ibma_dict = dict_from_paths(proc_paths, 119)

    if verbose:
        print(cbma_dict)
        print(ibma_dict)
    # exit()

    res_ALE, p_ALE = meta_ALE.fit(cbma_dict)
    res_KDA, p_KDA = meta_KDA.fit(cbma_dict)
    res_MKDA, p_MKDA = meta_MKDA.fit(cbma_dict)
    res_Stouffers, p_Stouffers = meta_Stouffers.fit(ibma_dict)
    res_Fishers, p_Fishers = meta_Fishers.fit(ibma_dict)

    # Threshold meta-analysis maps
    res_ALE, = fdr_threshold([res_ALE], p_ALE)
    res_KDA, = fdr_threshold([res_KDA], p_KDA)
    res_MKDA, = fdr_threshold([res_MKDA], p_MKDA)
    res_Stouffers, = fdr_threshold([res_Stouffers], p_Stouffers)
    res_Fishers, = fdr_threshold([res_Fishers], p_Fishers)

    # Turn thresholded maps into bool maps
    bool_ALE = img_utils.to_bool(res_ALE)
    bool_KDA = img_utils.to_bool(res_KDA)
    bool_MKDA = img_utils.to_bool(res_MKDA)
    bool_Stouffers = img_utils.to_bool(res_Stouffers)
    bool_Fishers = img_utils.to_bool(res_Fishers)

    # Compare meta-analysis result
    print(compare.kappa(bool_ALE, bool_KDA))
    print(compare.kappa(bool_ALE, bool_MKDA))
    print(compare.kappa(bool_KDA, bool_MKDA))
    print(compare.kappa(bool_ALE, bool_Stouffers))
    print(compare.kappa(bool_ALE, bool_Fishers))

    exit()

    plotting.plot_stat_map(res_ALE, title=f'{filename} ALE')
    plotting.plot_stat_map(res_KDA, title=f'{filename} KDA')
    plotting.plot_stat_map(res_MKDA, title=f'{filename} MKDA')

    plotting.plot_stat_map(res_Stouffers, title=f'{filename} Stouffers')
    plotting.plot_stat_map(res_Fishers, title=f'{filename} Fishers')
    plt.show()


if __name__ == '__main__':
    filename = 'hypo2_thresh.nii.gz'
    run_meta(filename, verbose=True)
