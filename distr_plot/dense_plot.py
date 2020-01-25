import distr_plot.plotter

import numpy as np
import matplotlib.pyplot as plt


def bins(array, segment_count):
    return (
        np.linspace(0, 1, 2*segment_count+1, endpoint=True)[1:-1:2],
        np.split(array, np.linspace(0, 1, segment_count+1, endpoint=True)[1:-1]),
    )
