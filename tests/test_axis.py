import numpy as np

from axis import project


def test_projection_sign_and_orientation():
    t = np.linspace(0, 1, 20)
    for xy in (np.c_[10 + 40 * t, 20 + 0 * t],            # run toward +x
               np.c_[50 - 40 * t, 30 + 0 * t],            # run toward -x
               np.c_[10 + 0 * t, 5 + 30 * t],             # run toward +y
               np.c_[10 + 30 * t, 5 + 30 * t]):           # diagonal
        p, ax = project(xy)
        assert p[-1] > p[0]
        assert np.isclose(np.linalg.norm(ax), 1.0)
        assert np.isclose(p[-1] - p[0], np.linalg.norm(xy[-1] - xy[0]))


def test_fixed_axis_keeps_sign():
    xy = np.c_[np.linspace(50, 10, 10), np.zeros(10)]
    p, ax = project(xy, axis=np.array([1.0, 0.0]))
    assert p[-1] < p[0]
