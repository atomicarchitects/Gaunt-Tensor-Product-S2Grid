import torch
import torch.nn as nn

import einops
from e3nn import o3
from .efficient_utils import FFT_batch_channel, sh2f_batch_channel, f2sh_batch_channel
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
sh2f_bases_dict = torch.load(os.path.join(current_directory,"coefficient_sh2f.pt"))
f2sh_bases_dict = torch.load(os.path.join(current_directory,"coefficient_f2sh.pt"))


class EfficientMultiTensorProduct(nn.Module):

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: int,
        device: str,
    ) -> None:
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_channels = irreps_in.count((0, 1))
        del irreps_in, irreps_out
        self.correlation = correlation
        self.device = device
        print("bro", self.irreps_in, self.irreps_out, self.num_channels, self.correlation, num_elements)

        L_in = self.irreps_in.lmax + 1
        L_out = self.irreps_out.lmax + 1
        self.L_in = L_in
        self.L_out = L_out

        self.sh2f_bases = sh2f_bases_dict[L_in].to(device)
        lmaxs = torch.arange(2, correlation + 1) * (L_in - 1)
        self.f2sh_bases_list = list(map(lambda lmax: f2sh_bases_dict[lmax + 1].to(device), lmaxs.tolist()))
        self.offsets_st = lmaxs - L_out + 1
        self.offsets_ed = lmaxs + L_out

        def gen_mask(L):
            left_indices = torch.arange(L, device=device).view(1, -1)  
            right_indices = torch.arange(L - 2, -1, -1, device=device).view(1, -1)  
            column_indices = torch.cat((left_indices, right_indices), dim=1).repeat(L, 1)  
            row_indices = torch.arange(L, device=device).view(-1, 1).repeat(1, 2 * L - 1)  
            mask = torch.abs(column_indices - (L - 1)) <= row_indices  
            mask2D = (torch.ones(L, 2 * L - 1, device=device) * mask).to(bool)
            return mask2D.flatten()
        self.mask_i, self.mask_o = list(map(gen_mask, [L_in, L_out]))

        slices = 2 * torch.arange(L_out, device=device) + 1
        self.slices = slices.tolist()
        
        self.weights = nn.ParameterDict({})
        for i in range(1, correlation + 1):
            w = nn.Parameter(
                torch.randn(1, num_elements, self.num_channels, self.L_out)
            )
            self.weights[str(i)] = w
    
    def forward(self, atom_feat: torch.tensor, atom_type: torch.Tensor):
        print("irreps_in", self.irreps_in, self.irreps_in.dim)
        print("inputs", atom_feat.shape, atom_type.shape)

        # inputs are of shape:
        # atom_feat: (B, C, ir)
        # atom_type: (B, num_elements)

        # self.weights[i] are all of shape: (1, num_elements, C, num_output_irreps)
        # atom_type.unsqueeze(-1).unsqueeze(-1) is of shape: (B, num_elements, 1, 1)
        # The product of the two is of shape: (B, num_elements, C, num_output_irreps)
        # weights is of shape: (B, C, num_output_irreps, 1) after summing over num_elements
        # feat4D is of shape: (B, C, L_in, 2L_in-1), which gets sliced to (B, C, num_output_irreps, 2num_output_irreps-1)
    
        # Convert from 3D to 4D, so as to facilitate the implementation of Efficient Gaunt TP
        # The time taken by this convert step is minimal
        n_nodes = atom_feat.shape[0]
        feat3D = torch.zeros(n_nodes, self.num_channels, self.mask_i.shape[0], device=self.device)
        feat3D[:, :, self.mask_i] = atom_feat
        feat4D = feat3D.reshape(n_nodes, self.num_channels, self.L_in, -1) # (B, C, L_in, 2L_in-1)

        print(atom_type[:5])
        print("feat3D", feat3D.shape)
        print("feat4D", feat4D.shape)
        for w in self.weights.values():
            print(w.shape, atom_type.unsqueeze(-1).unsqueeze(-1).shape)
    
        # @T.C.: Perform Efficient Gaunt TP  
        weights = (self.weights["1"] * atom_type.unsqueeze(-1).unsqueeze(-1)).sum(1).unsqueeze(-1) # (B, C, L_out, 1)
        print("weights", (self.weights["1"] * atom_type.unsqueeze(-1).unsqueeze(-1)).shape, weights.shape)
        result = feat4D[:, :, :self.L_out, self.L_in-self.L_out:self.L_in+self.L_out-1] * weights
        
        fs_out = {}
        fs_out[1] = sh2f_batch_channel(feat4D, self.sh2f_bases)
        for nu in range(2, self.correlation + 1):
            if nu % 2 == 0:
                fs_out[nu] = FFT_batch_channel(fs_out[nu//2], fs_out[nu//2])
            else:
                fs_out[nu] = FFT_batch_channel(fs_out[nu//2], fs_out[nu//2 + 1])
            idx = nu - 2
            weights = (self.weights[str(nu)] * atom_type.unsqueeze(-1).unsqueeze(-1)).sum(1).unsqueeze(-1)
            result += weights * f2sh_batch_channel(fs_out[nu], self.f2sh_bases_list[idx]).real[:, :, :self.L_out, self.offsets_st[idx]:self.offsets_ed[idx]]
        

        # Convert from 4D to 2D, so as to match the original codebase
        # The time taken by this convert step is minimal
        if self.L_out == 1:
            return  result.squeeze()
        else:
            result3D_unfiltered = result.reshape(n_nodes, self.num_channels, -1)
            result3D = torch.zeros(n_nodes, self.num_channels, self.mask_o.shape[0], device=self.device)
            result3D = result3D_unfiltered[ :, :, self.mask_o]
            irreps = torch.split(result3D, self.slices, dim=-1)
            irreps_flatten = list(map(lambda x: x.flatten(start_dim=1), irreps))
            result2D = torch.cat(irreps_flatten, dim=-1)
            return result2D
        

class GauntTensorProductS2Grid(nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        res_beta: int,
        res_alpha: int,
    ):
        super().__init__()

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)

        self.to_s2grid = o3.ToS2Grid(
            lmax=self.irreps_out.lmax,
            res=(res_beta, res_alpha),
        )
        self.from_s2grid = o3.FromS2Grid(
            lmax=self.irreps_out.lmax,
            res=(res_beta, res_alpha),
        )
    
    def forward(self, input1, input2):
        print("inputs", input1.shape, input2.shape)
        input1_grid = self.to_s2grid(input1)
        input2_grid = self.to_s2grid(input2)
        print("inputs on grid", input1_grid.shape, input2_grid.shape)
        output_grid = input1_grid * input2_grid
        print("output on grid", output_grid.shape)
        output = self.from_s2grid(output_grid)
        return output


        

class EfficientMultiTensorProductGauntS2Grid(nn.Module):

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: int,
        device: str,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_channels = irreps_in.count((0, 1))
        del irreps_in, irreps_out
        self.correlation = correlation
        self.device = device

        L_in = self.irreps_in.lmax + 1
        L_out = self.irreps_out.lmax + 1
        self.L_in = L_in
        self.L_out = L_out

        self.weights = nn.ParameterDict({})
        self.weight_repeats = torch.concatenate([torch.as_tensor(ir.dim).repeat(mul) for mul, ir in self.irreps_out]).to(device)
        
        for i in range(1, correlation + 1):
            w = nn.Parameter(
                torch.randn(1, num_elements, self.num_channels, self.L_out)
            )
            self.weights[str(i)] = w

        self.gaunt_s2grid = GauntTensorProductS2Grid(
            irreps_in1=self.irreps_in,
            irreps_in2=self.irreps_in,
            irreps_out=self.irreps_out,
            res_beta=2 * L_out,
            res_alpha=2 * L_out,
        )
    
    def forward(self, atom_feat: torch.tensor, atom_type: torch.Tensor):
        # Perform Efficient Gaunt TP
        for nu in range(2, self.correlation + 1):
            # Compute weights for this iteration.
            weights = (self.weights[str(nu)] * atom_type.unsqueeze(-1).unsqueeze(-1)).sum(1)
            # The weights are of shape: (B, C, L_out)
            # Repeat the weights for each dimension of the output irreps.
            weights = weights.repeat_interleave(self.weight_repeats, dim=-1)
            # The weights are now of shape: (B, C, self.irreps_out.dim)

            # Mix in weights, and add to current result.
            result += weights * prod        

            # Perform product.
            prod = self.gaunt_s2grid(prod, atom_feat)

        return result.squeeze()


