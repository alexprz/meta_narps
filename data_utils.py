"""Implement functions to handle NARPS data as input."""

import os
import warnings
import nibabel
import numpy as np
import ntpath

from nilearn.image import load_img


def retrieve_imgs(dir_path, filename, excepted=[]):
    """
    Return a list of paths to images.

    This function is specific to the studied dataset structure.
    It browses all the folders inside the specified one and for each
    encountered folder, looks for the specified filename. If it exists,
    adds its path to the list.

    Args:
        dir_path (str): Path to the folder containing the studies folders
        filename (str): Name of the file to look for in each study's fodler.
        excepted (list): List of folders (string) to ignore when browsing
            data folder

    Returns:
        (list): List of absolute paths (string) to the found images.

    """
    # List of folders contained in dir_path folder
    Dir = [f'{dir_path}{dir}' for dir in next(os.walk(dir_path))[1]]
    try:
        # On some OS the root dict is also in the list, must be removed
        Dir.remove(dir_path)
    except ValueError:
        pass

    # file, ext = filename.split('.', 1)

    # paths = dict()
    paths = list()
    for dir in Dir:
        path = f'{os.path.abspath(dir)}/{filename}'  # Turn into absolute paths

        # Filter to keep only existing files
        if os.path.isfile(path):
            paths.append(path)

        else:
            name = os.path.basename(os.path.normpath(dir))  # Study name
            warnings.warn(f"Study folder '{name}' has no file '{filename}'.")

    return paths


def iterator_paths_imgs(paths):
    """Yield path and Nifti images from a path list."""
    for path in paths:
        try:
            img = load_img(path)
            if np.isnan(img.get_fdata()).any():
                warnings.warn(f'Img {path} contains Nan. Ignored.')
                # continue
            yield path, img

        except ValueError:
            warnings.warn(f'File {path} not found. Ignored.')

        except nibabel.filebasedimages.ImageFileError:
            warnings.warn(f'File {path} found but not supported. Ignored.')


def iterator_name_imgs(paths):
    """Yield folder's names and Nifti images from a path list."""
    for path, img in iterator_paths_imgs(paths):
        study_dir, _ = ntpath.split(path)  # Extract path of study
        name = os.path.basename(os.path.normpath(study_dir))
        yield name, img


def iterator_imgs(paths):
    """Yield Nifti images from a path list."""
    for _, img in iterator_paths_imgs(paths):
        yield img


if __name__ == '__main__':
    data_dir = 'data/orig/'
    filename = 'hypo1_thresh.nii.gz'

    paths = retrieve_imgs(data_dir, filename)

    iter_imgs = iterator_imgs(paths)

    for img in iter_imgs:
        print(img)
