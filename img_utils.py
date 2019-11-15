"""Tools to work on brain images."""

import nibabel as nib


def to_bool(img):
    """Convert input img to bool img."""
    if isinstance(img, nib.Nifti1Image):
        img = img.get_fdata()

    img[img != 0] = 1

    return img
