import utils

import numpy as np
import matplotlib.pyplot as plt


def distr(x, y_distr, max_quantiles=np.inf, **kwargs):
    """
    Plots a distribution along the y-axis as it changes over the x-axis.
    """

    def make_colors(splits):
        splits = splits-2
        return [(0, 0, 0, 0)] + [
            3*(0.75*np.abs(i/(splits/2) - 1),) + (1,)
            for i in range(splits)
        ] + [(0, 0, 0, 0)]

    distr_array = utils.make_stack_plot_array(y_distr, max_quantiles).T
    plt.stackplot(
        x, distr_array, colors=make_colors(distr_array.shape[0]), baseline='sym'
    )
