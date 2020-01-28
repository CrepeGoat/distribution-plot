import distr_plot.distr_plot

import numpy as np
import matplotlib.pyplot as plt


def bins(array, bin_count):
    """
    Partitions an array into equal-ish-width bins. Returns
    - the center quantile corresponding to each bin, and
    - the samples in each bin
    """
    return (
        np.linspace(0, 1, 2*bin_count+1, endpoint=True)[1:-1:2],
        np.split(array, np.linspace(
            0, len(array), bin_count+1, endpoint=True
        )[1:-1].astype(np.int64)),
    )


def dense(x, y, bin_count, max_quantiles=np.inf, **kwargs):
    avg_indices, bin_arrays = bins(y, bin_count)
    return distr_plot.distr_plot.distr(
        np.quantile(x, avg_indices),
        bin_arrays,
        max_quantiles, **kwargs
    )
