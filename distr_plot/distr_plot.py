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


def _make_stackplot_array(quantile_lines):
    """
    Generates from a 2D "jagged array" of sample distributions a rectangular
    array that can be directly used in a stack plot.
    """
    maxabsval = max(quantile_lines.max(), -quantile_lines.min())
    return np.diff(quantile_lines, axis=0, prepend=-maxabsval, append=maxabsval)


def _make_stackplot_colors(stack_count):
    """
    Generates a list of rgba color values for the distribution stack plot.
    """
    def color(q):
        """
        Produces a color for a given quantile.
        """
        dist_to_median = 2*np.abs(q - 0.5)
        return (0.75*dist_to_median + 0.125,)*3 + (1,)

    color_count = stack_count-1

    blank = (0, 0, 0, 0)

    return [blank] + [
        color((i+0.5) / color_count)
        for i in range(color_count)
    ] + [blank]


###############################################################################

def distr(x, y_distr, max_quantiles=np.inf, **kwargs):
    """
    Plots a distribution along the y-axis as it changes over the x-axis.
    """
    distr_array = _make_quantile_lines(y_distr, max_quantiles)

    plot_objs = []
    plot_objs.extend(plt.plot(x, distr_array[0], 'k', linewidth=0.3))
    plot_objs.extend(plt.plot(x, distr_array[-1], 'k', linewidth=0.3))
    
    ylim = plt.ylim()
    plot_objs.extend(plt.stackplot(
        x, _make_stackplot_array(distr_array),
        colors=_make_stackplot_colors(distr_array.shape[0]),
        baseline='sym',
        **kwargs
    ))
    plt.ylim(ylim)  # sets visible range to ignore whitespace from stackplot
