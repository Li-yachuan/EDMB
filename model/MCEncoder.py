'''
shared decoder
'''

from model.vmamba import Backbone_VSSM
from model.caformer import caformer_m36_384_in21ft1k
from model.caformer import caformer_s18_384_in21ft1k
from torch import nn
import torch
import math
import random
from torch.nn.functional import interpolate


class MIXENC(nn.Module):
    def __init__(self, Dulbrn=16,
                 ckpt="output-VM/0613-bsds-su/epoch-11-training-record/epoch-11-checkpoint.pth",
                 mamba_ckpt="model/vssm_small_0229_ckpt_epoch_222.pth"):
        super(MIXENC, self).__init__()

        self.encoder = Backbone_VSSM(
            pretrained=None,
            out_indices=(0, 1, 2),
            # out_indices=(0, 1, 2, 3),
            dims=96,
            # depths=(2, 2, 15, 2),
            depths=(2, 2, 15, 0),
            ssm_d_state=1,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            ssm_conv=3,
            ssm_conv_bias=False,
            forward_type="v05_noz",  # v3_noz,
            mlp_ratio=4.0,
            downsample_version="v3",
            patchembed_version="v2",
            drop_path_rate=0.3,
            Dulbrn=Dulbrn)
        self.load_pretrained(ckpt)
        self.encoder.eval()
        for k, v in self.encoder.named_parameters():
            v.requires_grad = False

        from model.caformer import convformer_b36_384_in21ft1k,convformer_m36_384_in21ft1k
        # self.local_encoder = convformer_b36_384_in21ft1k(pretrained=True)
        self.local_encoder = convformer_m36_384_in21ft1k(pretrained=True)

        self.out_channels = self.encoder.out_channels

        self.adapter = nn.ModuleList([
            nn.Conv2d(self.local_encoder.out_channels[0],self.out_channels[2],kernel_size=1),
            nn.Conv2d(self.local_encoder.out_channels[1],self.out_channels[3],kernel_size=1),
            nn.Conv2d(self.local_encoder.out_channels[2],self.out_channels[4],kernel_size=1)
        ])

    def load_pretrained(self, ckpt=None, key="state_dict"):
        if ckpt is not None:
            try:
                _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
                incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
                # print(incompatibleKeys)
            except Exception as e:
                print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):

        global_feat = self.encoder(x)

        local_feat = self.local_encoder(x)
        local_feat_adapt = []
        for lf,conv in zip(local_feat,self.adapter):
            local_feat_adapt.append(conv(lf))

        local_feat_adapt = global_feat[:2] + local_feat_adapt


        return global_feat, local_feat_adapt

    # def cat_patch(self, f00i, f01i, f10i, f11i, fg):
    #     return interpolate(
    #         torch.cat([torch.cat([f00i, f01i], dim=3), torch.cat([f10i, f11i], dim=3)], dim=2),
    #         size=fg.size()[2:],
    #         mode="bilinear")

    # def align(self, x1, x2):
    #     if x1.size() != x2.size():
    #         x2 = interpolate(x2, x1.size()[2:], mode="bilinear")
    #     return x2
