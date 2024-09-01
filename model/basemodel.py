from torch import nn
from model.utils import get_encoder, get_decoder, get_enhence
from torch.nn.functional import interpolate
import torch
from math import sqrt
import numpy as np


class Basemodel(nn.Module):
    def __init__(self, args):
        # encoder_name="caformer-m36",
        # decoder_name="unetp"):
        super(Basemodel, self).__init__()

        self.encoder = get_encoder(args.encoder, global_ckpt=args.global_ckpt)
        self.decoder = get_decoder(args.decoder, self.encoder.out_channels)

    def forward(self, x, label_style=None):
        _, _, H, W = x.size()
        features = self.encoder(x)
        out = self.decoder(features)
        if isinstance(out, torch.Tensor) and out.size()[2:] != (H, W):
            out = interpolate(out, (H, W), mode="bilinear")
        return out
