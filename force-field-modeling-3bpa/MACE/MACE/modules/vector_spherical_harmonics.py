from typing import Tuple, Callable

import torch
import torch.nn as nn
import e3nn


def get_change_of_basis_matrix(lmax: int) -> Tuple[torch.Tensor, e3nn.o3.Irreps]:
    tp = e3nn.o3.ReducedTensorProducts("ij", i="1o", j=e3nn.o3.Irreps.spherical_harmonics(lmax))
    return tp.change_of_basis, tp.irreps_out


class VectorSphericalHarmonics(nn.Module):
    """Barebones implementation of vector spherical harmonics."""

    def __init__(self, lmax: int, res_beta: int, res_alpha: int) -> None:
        super().__init__()

        self.lmax = lmax
        self.res_beta = res_beta
        self.res_alpha = res_alpha

        self.rtp, self.vsh_irreps = get_change_of_basis_matrix(lmax=lmax)
        self.to_s2grid = e3nn.o3.ToS2Grid(lmax=lmax, res=(res_beta, res_alpha))
        self.from_s2grid = e3nn.o3.FromS2Grid(lmax=lmax, res=(res_beta, res_alpha))

    def to_vector_signal(self, vsh_coeffs: torch.Tensor) -> torch.Tensor:
        xyz_coeffs = torch.einsum("ijk,...i->...jk", self.rtp, vsh_coeffs)
        vector_signal = self.to_s2grid(xyz_coeffs)
        return vector_signal

    def from_vector_signal(self, vector_signal: torch.Tensor) -> torch.Tensor:
        xyz_coeffs = self.from_s2grid(vector_signal)
        vsh_coeffs = torch.einsum("ijk,...jk->...i", self.rtp, xyz_coeffs)
        return vsh_coeffs