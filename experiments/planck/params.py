import numpy as np


class PCosmology:

    def __init__(
        self,
        omch2: float = 0.12,
        ombh2: float = 0.019,
        h: float = 0.7,
        ln_10_10_As: float = -3.045,
        ns: float = 0.965,
        w: float = -1.0,
    ) -> None:

        self._omch2 = omch2
        self._ombh2 = ombh2
        self._h = h
        self._ln_10_10_As = ln_10_10_As
        self._ns = ns
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
    def H0(self):
        return self._h * 100

    @property
    def ns(self):
        return self._ns

    @property
    def As(self):
        return np.exp(self._ln_10_10_As) * 1e-10

    @property
    def w(self):
        return self._w
