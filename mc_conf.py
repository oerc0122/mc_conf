from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from itertools import islice, product, starmap
from random import random

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

def generate_bounds(data: float, error_bar: float) -> tuple[float, float]:
    """
    Convert value, error_bar  data to upper-lower bounds.

    Parameters
    ----------
    data : float
        Best data.
    error_bar : float
        Expected error bar.

    Returns
    -------
    upper
        Error upper bound.
    lower
        Error lower bound.
    """
    return (data + error_bar, data - error_bar)

def _sample_bounds_mc(data: Sequence[tuple[float, float]]) -> Iterator[list[float]]:
    """
    Sample the data upper/lower bounds using a Monte-Carlo approach.

    Randomly pick from upper and lower bounds for each datum

    Parameters
    ----------
    data : Sequence[tuple[float, float]]
        Data as a series of upper/lower bound pairs.

    Yields
    ------
    list[float]
        Data sample.
    """
    while True:
        yield [elem[random() < 0.5] for elem in data]

def _sample_bounds_all(data: Sequence[tuple[float, float]]) -> Iterator[list[float]]:
    """
    Sample all pairs of upper/lower bound possibilities.

    Parameters
    ----------
    data : Sequence[tuple[float, float]]
        Data as a series of upper/lower bound pairs.

    Yields
    ------
    list[float]
        Data sample.
    """

    yield from product(*data)

def sample_bounds(data: Sequence[tuple[float, float]], n: int | None = None) -> Iterator[list[float]]:
    """
    Sample N Monte Carlo combinations of upper/lower bounds.

    Parameters
    ----------
    data : Sequence[tuple[float, float]]
        Data as a series of upper/lower bound pairs.
    n : int, optional
        Number of MC combinations to try.

    Yields
    ------
    list[float]
        Data sample.

    Notes
    -----
    If number of MC samples (`n`) is greater than number of total samples (or is `None`),
    sample all combinations, no repeats guaranteed.
    """
    if n is None or n >= 2**len(data):
        yield from _sample_bounds_all(data)
    else:
        yield from islice(_sample_bounds_mc(data), n)


def mc_confidence_interval(
        sample: NDArray[float],
        func: Callable[[float, ...], float],
        params: Sequence[float] | Sequence[tuple[float, float]],
        error_bars: Sequence[float] | None = None,
        n: int | None = None,
        *,
        calc_best: bool = False,
        **kwargs,
) -> tuple[NDArray[float], NDArray[float]] | tuple[NDArray[float], NDArray[float], NDArray[float]]:
    """
    Generate confidence intervals for a given dataset and function.

    Parameters
    ----------
    sample : NDArray[float]
        Sequence of data points to sample function at.
    func : Callable[[float, ...], float]
        Model function, mapping input point to output.
        f(sample, *params) -> float

        .. note::
            Must take same number of parameters as provided in `params`.

    params : Sequence[float] | Sequence[tuple[float, float]]
        Either:
            Best estimates for parameters.
        Or
            Data as a series of upper/lower bound pairs.
    error_bars : Sequence[float], optional
        If separate from parameters, list of sequence bounds.
    n : int, optional
        Number of MC samples to perform
    **kwargs : dict
        Fixed parameters to pass to function.
    calc_best : bool
        Whether to also calculate and return best outcome.

    Returns
    -------
    max_arr
        Maximum values sampled with bounds.
    min_arr
        Minimum values sampled with bounds.
    best
        "Best" outcome from provided parameters.

    Raises
    ------
    TypeError
         If conflicting parameters passed.

    Examples
    --------
    >>> data = [(1.0, 0.5), (2.0, 0.3), (1.0, 0.3)]
    >>> sample = np.linspace(-1.0, 1.0, 20)
    >>> func = lambda x, a, b, c: a*x**2 + b*x + c
    >>> best, max_, min_ = mc_confidence_interval(sample, func, params=data, n=50, calc_best=True)

    >>> max_, min_ = mc_confidence_interval(sample, func, params=data, n=50)

    >>> params = [1.0, 2.0, 1.0]
    >>> errors = [0.5, 0.3, 0.3]
    >>> best, max_, min_ = mc_confidence_interval(sample, func, params=params, error_bars=errors, n=50, calc_best=True)
    """

    if error_bars is not None and isinstance(params[0], tuple):
        raise TypeError("`params` contains error bars, must not pass `error_bars`")
    if error_bars is None and isinstance(params[0], float):
        raise TypeError("`params does not contain error bars, must pass `error_bars`")

    if error_bars is not None and isinstance(params[0], float):
        params = tuple(zip(params, error_bars))

    bounds = tuple(starmap(generate_bounds, params))

    max_arr = np.full(len(sample), -np.inf)
    min_arr = np.full(len(sample), np.inf)

    for args in sample_bounds(bounds, n):
        result = np.array([func(x, *args, **kwargs) for x in sample])
        max_arr = np.maximum(max_arr, result)
        min_arr = np.minimum(min_arr, result)

    if calc_best:
        best = np.array([func(x, *(p[0] for p in params), **kwargs) for x in sample])
        return best, max_arr, min_arr

    return max_arr, min_arr


def plot_confidence_interval(
        sample: NDArray[float],
        max_arr: NDArray[float],
        min_arr: NDArray[float],
        best: NDArray[float] | None = None,
        file_name: str | Path = None,
) -> plt.Axes:
    """
    Plot confidence interval as generated by `mc_confidence_interval`.

    Parameters
    ----------
    file_name : str | Path
        File to write figure to.
    sample : NDArray[float]
        Sampling points (x-axis).
    max_arr : NDArray[float]
        Calculated maximum points.
    min_arr : NDArray[float]
        Calculated minimum points.
    best : NDArray[float], optional
        Calculated best fit.

    Returns
    -------
    matplotlib.pyplot.axes
        Axes with plotted data and interval.
    """
    if file_name is not None:
        fig = plt.figure()
        ax = plt.axes()
        fig.add_axes(ax)
    else:
        ax = plt.axes()

    ax.fill_between(sample, min_arr, max_arr)
    ax.plot(sample, min_arr, color='b')
    ax.plot(sample, max_arr, color='b')
    if best is not None:
        ax.plot(sample, best, color='r')

    if file_name is not None:
        fig.savefig(file_name)

    return ax


if __name__ == "__main__":
    #        1 ± 0.5  ,  2 ± 0.3  ,  1 ± 0.3
    data = [(1.0, 0.5), (2.0, 0.3), (1.0, 0.3)]

    sample = np.linspace(-1.0, 1.0, 20)
    func = lambda x, a, b, c: a*x**2 + b*x + c
    best, *max_min = mc_confidence_interval(sample, func, params=data, n=50, calc_best=True)

    plot_confidence_interval(sample, *max_min, best, file_name='my_plot.png')
