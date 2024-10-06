
import math
import torch
import os
import collections


import torch.nn as nn
import torch.nn.functional as F

from torch import nn, einsum, Tensor

from einops import rearrange, pack, unpack


class SoftMoELayer(nn.Module):
    def __init__(self, experts, scales, d, n, d2):
        super(SoftMoELayer, self).__init__()
        self.Phi_1 = nn.Linear(d, n)
        self.face_scale = nn.Linear(d2, 1)
        self.face_pool = nn.AdaptiveAvgPool1d(1)
        self.hand_scale = nn.Linear(d2, 1)
        self.hand_pool = nn.AdaptiveAvgPool1d(1)
        self.experts = experts
        self.scales = scales

    def forward(self, X):
        is_conv = False

        if len(X.size())==4:
            """
            This is for conv lora
            """
            is_conv = True
            B, d, H, W = X.size()
            X = X.permute(0, 2, 3, 1).view(B, -1, d)

        logits = self.Phi_1(X) 
        B, m, n = logits.size()

        D = torch.sigmoid(logits) # b m n

        results = [f_i(X.permute(0, 2, 1).view(B, d, int(math.sqrt(m)), -1) if is_conv else X) * self.scales[i] for i, f_i in enumerate (self.experts)]
        

        for ind, (result, pool, scale_module) in enumerate(zip(results, [self.face_pool, self.hand_pool], [self.face_scale, self.hand_scale])):
            if is_conv:
                # print(result.view(B, d, -1).size(), scale_module.weight.shape, (pool(result.view(B, d, -1)).size()))
                results[ind] = result.permute(0, 2, 3, 1).view(B, -1, d) * D[:,:,ind].unsqueeze(dim=2) * torch.sigmoid(scale_module(pool(result.view(B, d, -1)).permute(0, 2, 1)))  ## soft assign 
            else:
                # print(result.permute(0, 2, 1).size(), scale_module.weight.shape, torch.sigmoid(scale_module(pool(result.view(B, d, -1)))).size())
                results[ind] = result * D[:,:,ind].unsqueeze(dim=2) * torch.sigmoid(scale_module(pool(result.permute(0, 2, 1)).permute(0, 2, 1))) # soft assign
            
        ## cal scale weight
        # Y = torch.cat(results, dim=2)
        # # print("Y size:", Y.shape)
        # Y = self.Phi_2(Y)  ## b m 2*d -> b m d 

        if is_conv:
            return results[0].permute(0, 2, 1).view(B, d, H, W) + results[1].permute(0, 2, 1).view(B, d, H, W)
        else:
            return results[0] + results[1]


class WrapperMoELayer(nn.Module):
    def __init__(self, experts, name, n):

        """
        experts: a list of experts, e.g., experts = [face, hand], they are instances of LoRAModule
        """
        super().__init__()

        ## prepare experts
        experts_remove_org = [nn.Sequential(collections.OrderedDict([('lora_down', expert.lora_down), 
                                                            ('lora_up', expert.lora_up)])) for expert in experts]

        try:
            d = experts[0].lora_down.in_features
            d2 = experts[0].lora_up.out_features
        except: 
            d = experts[0].lora_down.in_channels
            d2 = experts[0].lora_up.out_channels

        scales = [expert.scale for expert in experts]

        self.softmoe = SoftMoELayer(experts_remove_org, scales, d, n, d2)
        self.moe_name = "moe_" + name

        self.expert = experts[0]
        self.main_branch = self.expert.org_module.forward  ## must before the applyto function of LoRAModule
        self.expert.org_module.forward = self.forward
        del self.expert

    def forward(self, x):
        
        Y = self.softmoe(x)
        
        x = self.main_branch(x)

        # print(self.moe_name, self.softmoe.experts[1].lora_down.weight.requires_grad, self.softmoe.experts[1].lora_down.weight)

        return x + Y  # * self.gamma 
  
import os 

class WrapperLoRANetwork(nn.Module):
    def __init__(self, networks_list, num_experts):
        super().__init__()
        """
        netowrks_list: List of networks, each network contains vast of LoRAModules

        default we have two experts [face, hand]
        """
        assert len(networks_list) == num_experts, "the length of the networks_list should be equal to the number of experts"

        num_LoRAModules = len(networks_list[0].unet_loras)

        self.moelayer_list = []
        for (faceLoRAModule, handLoRAModule) in zip(networks_list[0].unet_loras, networks_list[1].unet_loras):

            self.moelayer_list.append(WrapperMoELayer([faceLoRAModule, handLoRAModule], handLoRAModule.lora_name, num_experts))

        print(f"init {len(self.moelayer_list)} MoE layer")

    def prepare_optimizer_params(self):
        self.requires_grad_(True)
        all_params = []
        def enumerate_params(moelayers):
            params = []
            for moelayer in moelayers:
                params.extend(moelayer.parameters())
            return params

        param_data = {"params": enumerate_params(self.moelayer_list)}

        all_params.append(param_data)
        return all_params
    
    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def get_trainable_params(self):
        return self.parameters()

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)

    def apply_to(self):
        for moelayer in self.moelayer_list:
            self.add_module(moelayer.moe_name, moelayer)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)


    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

def create_moe(networks_list, num_experts=2):
    moe_network = WrapperLoRANetwork(networks_list, num_experts)
    return moe_network



            


