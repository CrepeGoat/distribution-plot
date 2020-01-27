import distr_plot.utils

import numpy as np
import matplotlib.pyplot as plt


def distr(x, y_distr, max_quantiles=np.inf, **kwargs):
    """
    Plots a distribution along the y-axis as it changes over the x-axis.
    """

    def make_colors(splits):
        splits = splits-1
        return [(0, 0, 0, 0)] + [
            3*(0.75*np.abs(i/(splits/2) - 1),) + (1,)
            for i in range(splits)
        ] + [(0, 0, 0, 0)]

    distr_array = distr_plot.utils.make_quantile_lines(y_distr, max_quantiles)

    plot_objs = []
    plot_objs.extend(plt.plot(x, distr_array[0], '-'))
    plot_objs.extend(plt.plot(x, distr_array[-1], '-'))
    
    ylim = plt.ylim()
    plot_objs.extend(plt.stackplot(
        x, distr_plot.utils.make_stack_plot_array(distr_array),
        colors=make_colors(distr_array.shape[0]),
        baseline='sym',
        **kwargs
    ))
    plt.ylim(ylim)  # sets visible range to ignore whitespace from stackplot
