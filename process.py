"""Functions used to process the raw data."""

import os
from nilearn import datasets, image, plotting

import matplotlib
from matplotlib import pyplot as plt


matplotlib.use('MacOSX')

template = datasets.load_mni152_template()


def process_one_img(i_img):
    """Process raw image."""
    return image.resample_to_img(i_img, template)


def process(iter_names_imgs, o_dir, o_filename, skip_exist=True, verbose=False):
    """
    Process raw images.

    Args:
        iter_names_imgs (sequence): Sequence of size 2 tuple (name, img).
        o_dir (string): Path to the output directory
        o_filename (string): Filename of the processed image.
        skip_exist (bool): If True, ignore if the output file already exist.

    """
    o_dir = os.path.abspath(o_dir)  # Turn into absolute paths
    os.makedirs(o_dir, exist_ok=True)

    for name, i_img in iter_names_imgs:
        if verbose:
            print(f'Processing {name}...')
        o_study_dir = f'{o_dir}/{name}/'
        os.makedirs(o_study_dir, exist_ok=True)
        o_path = f'{o_study_dir}{o_filename}'

        # plotting.plot_stat_map(i_img, title=f'{name}')
        # plt.show()

        if skip_exist and os.path.isfile(o_path):
            continue

        o_img = process_one_img(i_img)
        o_img.to_filename(o_path)

    # plt.show()
