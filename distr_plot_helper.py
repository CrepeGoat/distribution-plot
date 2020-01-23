import numpy as np


def jagged_index_quantiles(jagged, quantiles):
    """
    Takes a rectangular slice of a "jagged array". The quantiles dictate the
    element positions in each jagged subarray, as a percentage of the full
    subarray length. For each quantile and jagged subarray, the nearest
    element in the subarray to the given quantile is used.
    """
    result = np.zeros((len(jagged), len(quantiles)))

    for i, subarray in enumerate(jagged):
        result[i] = np.quantile(subarray, quantiles)

    return result


def spaced_quantiles(n):
    """
    Generates an array of evenly-spaced quantiles.
    """
    return np.linspace(0, 1, num=n, endpoint=True)


def make_stack_plot_array(distr, axis=0, max_quantiles=np.inf):
    """
    Generates from a 2D array of sample distributions an array that can be used
    to plot distribution data in a stack plot.
    """
    maxabsval = max(distr.max(), -distr.min())

    sorted_array = np.sort(distr, axis=axis)
    if len(sorted_array) > max_quantiles:
        sorted_array = sorted_array.take(
            np.linspace(0, len(sorted_array)-1, num=max_quantiles).astype(np.int),
            axis=axis
        )

    return np.diff(sorted_array, axis=axis, prepend=-maxabsval, append=maxabsval)
