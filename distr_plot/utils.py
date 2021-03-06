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
        result[i] = np.quantile(subarray, quantiles, interpolation='nearest')

    return result


def spaced_quantiles(n):
    """
    Generates an array of evenly-spaced quantiles.
    """
    return np.linspace(0, 1, num=n, endpoint=True)


def make_quantile_lines(distr, max_quantiles=np.inf):
    """
    Generates from a 2D "jagged array" of sample distributions a rectangular
    array of quantile lines.
    """
    jagged_maxlen = max(len(a) for a in distr)
    quantiles_count = min(max_quantiles, jagged_maxlen)

    return jagged_index_quantiles(
        distr, quantiles=spaced_quantiles(quantiles_count)
    ).T


def make_stack_plot_array(quantile_lines):
    """
    Generates from a 2D "jagged array" of sample distributions a rectangular
    array that can be directly used in a stack plot.
    """
    maxabsval = max(quantile_lines.max(), -quantile_lines.min())
    return np.diff(quantile_lines, axis=0, prepend=-maxabsval, append=maxabsval)
