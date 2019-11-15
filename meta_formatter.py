"""Put data in the right format to be read as meta-analysis input."""
import numpy as np


def get_sub_dict(XYZ, path_dict, sample_size):
    """
    Build sub dictionnary of a study using the nimare structure.

    Args:
        XYZ (tuple): Size 3 tuple of list storing the X Y Z coordinates.
        path_dict (dict): Dict which has map name ('t', 'z', 'con', 'se')
            as keys and absolute path to the image as values.
        sample_size (int): Size of the sample.

    Returns:
        (dict): Dictionary storing the coordinates for a
            single study using the Nimare structure.

    """
    d = {
        'contrasts': {
            '0': {
                'metadata': {'sample_sizes': sample_size}
            }
        }
    }

    if XYZ is None:
        XYZ = [0], [0], [0]

    if XYZ is not None and len(XYZ[0]) > 0:
        d['contrasts']['0']['coords'] = {
                    'x': XYZ[0],
                    'y': XYZ[1],
                    'z': XYZ[2],
                    'space': 'MNI'
                    }
        d['contrasts']['0']['sample_sizes'] = sample_size

    if path_dict is not None:
        if isinstance(path_dict, str):
            path_dict = {'z': path_dict}

        d['contrasts']['0']['images'] = path_dict

    return d


def dict_from_activations(activations, sample_sizes):
    sub_dicts = []

    if isinstance(sample_sizes, int):
        sample_sizes = sample_sizes*np.ones(len(activations))

    for i, XYZ in enumerate(activations):
        sub_dict = get_sub_dict(XYZ, None, sample_sizes[i])
        sub_dicts.append(sub_dict)

    return {k: v for k, v in enumerate(sub_dicts)}


def dict_from_paths(paths, sample_sizes):
    sub_dicts = []

    if isinstance(sample_sizes, int):
        sample_sizes = sample_sizes*np.ones(len(paths))

    for i, path in enumerate(paths):
        sub_dict = get_sub_dict(None, path, sample_sizes[i])
        sub_dicts.append(sub_dict)

    return {k: v for k, v in enumerate(sub_dicts)}
