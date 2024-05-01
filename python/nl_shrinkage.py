import numpy as np
from numpy import matlib as mtl
import matlab.engine


def nl_shrinkage(samples):
    N, p = samples.shape
    sample_cov = samples.T @ samples / N
    values, vectors = np.linalg.eig(sample_cov)
    sort_idx = np.argsort(values)
    values = values[sort_idx]
    vectors = vectors[:, sort_idx]
    values = np.real(values[np.maximum(0, p - N):])
    vectors = np.real(vectors)
    L = mtl.repmat(values[..., np.newaxis], 1, np.minimum(N, p))
    h = N**(-1 / 3)
    H = h * L.T
    x = (L - L.T) / H
    f_tilde = 3 / (4 * np.sqrt(5)) * np.mean(np.maximum(1 - x**2 / 5, 0) / H, 1)
    Hftemp = -3 / (10 * np.pi) * x + 3 / (4 * np.sqrt(5) * np.pi) * (1 - x**2 / 5) * np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
    np.putmask(Hftemp, np.abs(x) == np.sqrt(5), -3 / (10 * np.pi) * x)
    H_ftilde = np.mean(Hftemp / H, axis=1)
    if p <= N:
        d_tilde = values / ((np.pi * p / N * values * f_tilde)**2 + (1 - p / N - np.pi * p / N * values * H_ftilde)**2)
    else:
        H_ftilde0 = (1 / np.pi) * (3 / (10 * h**2)
                                   + 3 / (4 * np.sqrt(5) * h) * (1 - 1 / (5 * h**2)) * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))) * \
                                   np.mean(1 / values)
        d_tilde0 = 1 / (np.pi * (p - N) / N * H_ftilde0)
        d_tilde1 = values / (np.pi**2 * values**2 * (f_tilde**2 + H_ftilde**2))
        d_tilde = np.hstack((d_tilde0 * np.ones(p - N), d_tilde1))
    sigmatilde = vectors @ np.diag(d_tilde) @ vectors.T
    return sigmatilde


class MatlabShrinkage:
    def __init__(self):
        self.eng = matlab.engine.start_matlab()

    def numpy_to_matlab(self, array):
        return matlab.double(array.tolist())

    def nl_shrinkage(self, samples):
        N, p = samples.shape
        sample_cov = samples.T @ samples / N
        return np.asarray(self.eng.analytical_shrinkage(self.numpy_to_matlab(sample_cov), float(N), float(p)))

    def nl_shrinkage_cov(self, cov, n, p):
        return np.asarray(self.eng.analytical_shrinkage(self.numpy_to_matlab(cov), float(n), float(p)))
