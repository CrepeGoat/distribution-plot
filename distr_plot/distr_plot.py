import numpy as np
import matplotlib.pyplot as plt


def _jagged_index_quantiles(jagged, quantiles):
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


def _spaced_quantiles(n):
    """
    Generates an array of evenly-spaced quantiles.
    """
    return np.linspace(0, 1, num=n, endpoint=True)


def _make_quantile_lines(distr, max_quantiles=np.inf):
    """
    Generates from a 2D "jagged array" of sample distributions a rectangular
    array of quantile lines.
    """
    jagged_maxlen = max(len(a) for a in distr)
    quantiles_count = min(max_quantiles, jagged_maxlen)

    return _jagged_index_quantiles(
        distr, quantiles=_spaced_quantiles(quantiles_count)
    ).T


def _make_stack_plot_array(quantile_lines):
    """
    Generates from a 2D "jagged array" of sample distributions a rectangular
    array that can be directly used in a stack plot.
    """
    maxabsval = max(quantile_lines.max(), -quantile_lines.min())
    return np.diff(quantile_lines, axis=0, prepend=-maxabsval, append=maxabsval)


def _make_stack_colors(splits):
    """
    Generates a list of rgba color values for the distribution stack plot.
    """
    splits = splits-1
    return [(0, 0, 0, 0)] + [
        3*(0.75*np.abs(i/(splits/2) - 1),) + (1,)
        for i in range(splits)
    ] + [(0, 0, 0, 0)]


###############################################################################

def distr(x, y_distr, max_quantiles=np.inf, **kwargs):
    """
    Plots a distribution along the y-axis as it changes over the x-axis.
    """
    distr_array = _make_quantile_lines(y_distr, max_quantiles)

    plot_objs = []
    plot_objs.extend(plt.plot(x, distr_array[0], '-'))
    plot_objs.extend(plt.plot(x, distr_array[-1], '-'))
    
    ylim = plt.ylim()
    plot_objs.extend(plt.stackplot(
        x, _make_stack_plot_array(distr_array),
        colors=_make_stack_colors(distr_array.shape[0]),
        baseline='sym',
        **kwargs
    ))
    plt.ylim(ylim)  # sets visible range to ignore whitespace from stackplot
