import numpy as np


class MOPEDstore:
    def __init__(self, b_matrix: np.ndarray, ycomp: np.ndarray):
        self._b_matrix = b_matrix
        self._ycomp = ycomp

    @property
    def b_matrix(self):
        return self._b_matrix

    @property
    def ycomp(self):
        return self._ycomp


def vectors(grad, covariance):

    ndata, ndim = grad.shape
    sol = np.linalg.inv(covariance) @ grad
    moped_vectors = np.zeros_like(sol)

    for i in range(ndim):
        # the first MOPED vector easy to compute
        if i == 0:
            moped_vectors[:, i] = sol[:, 0] / np.sqrt(grad[:, 0].T @ sol[:, 0])

        else:
            # create empty matrices to store pre-computations for the MOPED vectors
            dum_num = np.zeros((ndata, i))
            dum_den = np.zeros((i))

            for j in range(i):
                dum_num[:, j] = (grad[:, i].T @ moped_vectors[:, j]) * moped_vectors[
                    :, j
                ]
                dum_den[j] = (grad[:, i].T @ moped_vectors[:, j]) ** 2

            # the numerator
            moped_num = sol[:, i] - np.sum(dum_num, axis=1)

            # the denominator term
            moped_den = np.sqrt(grad[:, i].T @ sol[:, i] - np.sum(dum_den))

            # the MOPED vector
            moped_vectors[:, i] = moped_num / moped_den

    # check we are doing everything right
    for i in range(ndim):
        for j in range(i + 1):
            prod = moped_vectors[:, i].T @ covariance @ moped_vectors[:, j]
            print(f"{i} {j} : {prod.item():7.5f}")
    return moped_vectors
