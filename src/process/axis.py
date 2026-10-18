"""Run-axis projection shared by the forty and shuttle stages."""
import numpy as np


def principal_axis(xy: np.ndarray) -> np.ndarray:
    """Unit vector of the first principal component of an (n, 2) array of positions."""
    c = xy - xy.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(c.T))
    return vecs[:, -1]


def project(xy: np.ndarray, axis: np.ndarray | None = None, positive_run: bool = True):
    """Project (n, 2) positions to 1D position along the principal axis.

    With positive_run, the axis sign is chosen so the final position exceeds the first
    (forty: the run is positive). Returns (p, axis).
    """
    ax = principal_axis(xy) if axis is None else axis
    p = (xy - xy.mean(axis=0)) @ ax
    if axis is None and positive_run and p[-1] < p[0]:
        ax, p = -ax, -p
    return p, ax
