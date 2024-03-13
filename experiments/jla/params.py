"""
Code: JLA Cosmology representation.
Date: March 2024
Author: Arrykrishna
"""


class Cosmology:

    def __init__(
        self, omch2: float = 0.12, ombh2: float = 0.019, h: float = 0.7, w: float = -1.0
    ) -> None:
        self._omch2 = omch2
        self._ombh2 = ombh2
        self._h = h
        self._w = w

    @property
    def ombh2(self):
        return self._ombh2

    @property
    def omch2(self):
        return self._omch2

    @property
    def Omega_b(self):
        return self._omega_b / self._h**2

    @property
    def Omega_c(self):
        return self._omega_c / self._h**2

    @property
    def Omega_m(self):
        return (self._ombh2 + self._omch2) / self._h**2

    @property
    def h(self):
        return self._h

    @property
    def w(self):
        return self._w


class Nuisance:
    def __init__(self, Mb: float, delta_M: float, alpha: float, beta: float) -> None:

        self._Mb = Mb
        self._delta_M = delta_M
        self._alpha = alpha
        self._beta = beta

    @property
    def Mb(self):
        return self._Mb

    @property
    def delta_M(self):
        return self._delta_M

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta
