import numpy as np


def resize2D_to_rectangular(jagged):
    result = np.zeros((len(jagged), max(len(a) for a in jagged)))
    new_item_positions_on_axis = np.linspace(0, 1, num=(2*result.shape[1]+1))[1:-1:2]

    for i, a in enumerate(jagged):
        result[i] = np.asarray(a)[np.digitize(
            new_item_positions_on_axis,
            np.linspace(0, 1, num=(len(a)+1))[1:-1],
        )]

    return result


def make_wave_dist_array(array, axis=0, max_percentiles=np.inf):
    maxabsval = max(array.max(), -array.min())

    sorted_array = np.sort(array, axis=axis)
    if len(sorted_array) > max_percentiles:
        sorted_array = sorted_array.take(
            np.linspace(0, len(sorted_array)-1, num=max_percentiles).astype(np.int),
            axis=axis
        )

    return np.diff(sorted_array, axis=axis, prepend=-maxabsval, append=maxabsval)
