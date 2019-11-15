"""Compare meta-analysis results."""

from sklearn.metrics import confusion_matrix
import numpy as np
import nibabel as nib

# def _kappa(a, b, c, d):
#     s = a+b+c+d
#     x = ((a+c)*(a+b) + (c+d)*(b+d))

#     return 1. - s*(b+c)/(s**2-x)


def _kappa(a, b, c, d):
    s = a+b+c+d
    p_o = (a+d)/s
    p_e = ((a+c)*(a+b) + (c+d)*(b+d))/(s**2)

    return (p_o - p_e)/(1 - p_e)


def kappa(arr1, arr2):
    """Compute Cohen's kappa from two bool arrays."""
    if isinstance(arr1, nib.Nifti1Image):
        arr1 = arr1.get_fdata()
    if isinstance(arr2, nib.Nifti1Image):
        arr2 = arr2.get_fdata()

    M = confusion_matrix(arr1.ravel(), arr2.ravel())
    a = M[0, 0]
    b = M[0, 1]
    c = M[1, 0]
    d = M[1, 1]

    return _kappa(a, b, c, d)


if __name__ == '__main__':
    A = np.array([0, 0, 1, 1])
    B = np.array([1, 1, 0, 0])

    print(kappa(A, B))
