"""Rauch-Tung-Striebel smoother with a constant-acceleration (white-jerk) motion model.

State per coordinate is [position, velocity, acceleration]; x and y are smoothed independently with the
same parameters, so the 2D position/velocity/acceleration variances are the sum of per-axis contributions.
"""
import numpy as np


def _transition(dt: float, q: float):
    F = np.array([[1.0, dt, dt**2 / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    Q = q * np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6],
                      [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                      [dt**3 / 6, dt**2 / 2, dt]])
    return F, Q


def rts_smooth_1d(z: np.ndarray, t: np.ndarray, sigma: float, q: float, v0: float = 0.0):
    """Smooth a 1D position series. Returns (states (n, 3), covariances (n, 3, 3)).

    sigma: measurement SD (yd); q: jerk spectral density (yd^2/s^5); v0: prior mean velocity at the first frame.
    """
    n = len(z)
    H = np.array([[1.0, 0.0, 0.0]])
    R = sigma**2
    xp, Pp = np.zeros((n, 3)), np.zeros((n, 3, 3))   # predicted
    xf, Pf = np.zeros((n, 3)), np.zeros((n, 3, 3))   # filtered
    x = np.array([z[0], v0, 0.0])
    P = np.diag([sigma**2, 25.0, 100.0])
    for k in range(n):
        if k > 0:
            F, Q = _transition(float(t[k] - t[k - 1]), q)
            x, P = F @ xf[k - 1], F @ Pf[k - 1] @ F.T + Q
        xp[k], Pp[k] = x, P
        S = (H @ P @ H.T)[0, 0] + R
        K = (P @ H.T)[:, 0] / S
        x = x + K * (z[k] - x[0])
        P = (np.eye(3) - np.outer(K, H[0])) @ P
        xf[k], Pf[k] = x, 0.5 * (P + P.T)
    xs, Ps = xf.copy(), Pf.copy()
    for k in range(n - 2, -1, -1):
        F, _ = _transition(float(t[k + 1] - t[k]), q)
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs, Ps


def smooth_xy(xy: np.ndarray, t: np.ndarray, axis: np.ndarray, sigma: float, q: float) -> dict:
    """Smooth (n, 2) positions and project onto `axis` (unit vector).

    Returns signed p, v, a along the axis with variances, plus the smoothed 2D speed `speed`.
    """
    sx, Px = rts_smooth_1d(xy[:, 0], t, sigma, q)
    sy, Py = rts_smooth_1d(xy[:, 1], t, sigma, q)
    out = {}
    for i, name in enumerate(["p", "v", "a"]):
        out[name] = axis[0] * sx[:, i] + axis[1] * sy[:, i]
        out[f"{name}_var"] = axis[0] ** 2 * Px[:, i, i] + axis[1] ** 2 * Py[:, i, i]
    out["speed"] = np.hypot(sx[:, 1], sy[:, 1])
    return out
