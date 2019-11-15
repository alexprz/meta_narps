"""Extract coordinates from images."""

import nilearn
from nilearn import image, masking
from nipy.labs.statistical_mapping import get_3d_peaks

from data_utils import iterator_imgs, retrieve_imgs


template = nilearn.datasets.load_mni152_template()
gray_mask = masking.compute_gray_matter_mask(template)


def get_activations_one_img(img, threshold, verbose=False):
    """
    Retrieve the xyz activation coordinates from an image.

    Args:
        img (Nifti1Image): Nifti1Image from which to extract coordinates.
        threshold (float): value below threshold are ignored. Used for
            peak detection.

    Returns:
        (tuple): Size 3 tuple of lists storing respectively the X, Y and
            Z coordinates.

    """
    if verbose:
        print('Extracting')
    X, Y, Z = [], [], []

    img = image.resample_to_img(img, template)

    peaks = get_3d_peaks(img, mask=gray_mask, threshold=threshold)

    if not peaks:
        return X, Y, Z

    for peak in peaks:
        X.append(peak['pos'][0])
        Y.append(peak['pos'][1])
        Z.append(peak['pos'][2])

    del peaks
    return X, Y, Z


def get_activations(imgs, threshold, verbose=False):
    return [get_activations_one_img(img, threshold, verbose) for img in imgs]


if __name__ == '__main__':
    data_dir = 'data/orig/'
    filename = 'hypo1_thresh.nii.gz'

    paths = retrieve_imgs(data_dir, filename)
    iter_imgs = iterator_imgs(paths[:3])

    print(get_activations(iter_imgs, 1.96))
