import distr_plot.plotter

import numpy as np
import matplotlib.pyplot as plt


def bins(array, segment_count):
    """
    Partitions an array into equal-ish-width bins. Returns
    - the center quantile corresponding to each bin, and
    - the samples in each bin
    """
    return (
        np.linspace(0, 1, 2*segment_count+1, endpoint=True)[1:-1:2],
        np.split(array, np.linspace(0, 1, segment_count+1, endpoint=True)[1:-1]),
    )


def dense_plot(x, y, segment_count, max_quantiles=np.inf, **kwargs):
    avg_indices, bin_arrays = bins(y, segment_count)
    distr_plot.plotter.distr(
        np.quantile(x, avg_indices),
        bin_arrays,
        max_quantiles, **kwargs
    )
