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

        self.local_model = Backbone_VSSM(
            pretrained=mamba_ckpt,
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
            Dulbrn=0)

        self.out_channels = self.encoder.out_channels

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

        x = interpolate(x, scale_factor=2, mode="bilinear")
        _, _, H, W = x.size()

        grad = random.randint(0, 4)
        if grad == 0:
            feat_00 = self.local_model(x[..., :H // 2 + 1, :W // 2 + 1])
            with torch.no_grad():
                feat_01 = self.local_model(x[..., :H // 2 + 1, W // 2 + 1:])
                feat_10 = self.local_model(x[..., H // 2 + 1:, :W // 2 + 1])
                feat_11 = self.local_model(x[..., H // 2 + 1:, W // 2 + 1:])
        elif grad == 1:
            feat_01 = self.local_model(x[..., :H // 2 + 1, W // 2 + 1:])
            with torch.no_grad():
                feat_00 = self.local_model(x[..., :H // 2 + 1, :W // 2 + 1])
                feat_10 = self.local_model(x[..., H // 2 + 1:, :W // 2 + 1])
                feat_11 = self.local_model(x[..., H // 2 + 1:, W // 2 + 1:])
        elif grad == 2:
            feat_10 = self.local_model(x[..., H // 2 + 1:, :W // 2 + 1])
            with torch.no_grad():
                feat_00 = self.local_model(x[..., :H // 2 + 1, :W // 2 + 1])
                feat_01 = self.local_model(x[..., :H // 2 + 1, W // 2 + 1:])
                feat_11 = self.local_model(x[..., H // 2 + 1:, W // 2 + 1:])
        else:
            feat_11 = self.local_model(x[..., H // 2 + 1:, W // 2 + 1:])
            with torch.no_grad():
                feat_00 = self.local_model(x[..., :H // 2 + 1, :W // 2 + 1])
                feat_01 = self.local_model(x[..., :H // 2 + 1, W // 2 + 1:])
                feat_10 = self.local_model(x[..., H // 2 + 1:, :W // 2 + 1])

        local_feat = [self.cat_patch(f00i, f01i, f10i, f11i, fg)
                      for f00i, f01i, f10i, f11i, fg in
                      zip(feat_00, feat_01, feat_10, feat_11, global_feat)]
        local_feat = global_feat[:2] + local_feat
        return global_feat, local_feat

    def cat_patch(self, f00i, f01i, f10i, f11i, fg):
        return interpolate(
            torch.cat([torch.cat([f00i, f01i], dim=3), torch.cat([f10i, f11i], dim=3)], dim=2),
            size=fg.size()[2:],
            mode="bilinear")

    # def align(self, x1, x2):
    #     if x1.size() != x2.size():
    #         x2 = interpolate(x2, x1.size()[2:], mode="bilinear")
    #     return x2
