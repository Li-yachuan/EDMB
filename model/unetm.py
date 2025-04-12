from torch import nn
import torch
from torch.nn import functional as F

from model.unetp_mamba import DecoderBlock_vss, DecoderBlock_ir


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
    ):
        super().__init__()

        self.depth = len(encoder_channels)
        convs = dict()
        for d in range(self.depth - 1):
            DecoderBlock = DecoderBlock_ir if d == 0 else DecoderBlock_vss
            if d == self.depth - 2:
                convs["conv{}".format(d)] = DecoderBlock(encoder_channels[d + 1],
                                                         encoder_channels[d],
                                                         decoder_channels[d])
            else:
                convs["conv{}".format(d)] = DecoderBlock(decoder_channels[d + 1],
                                                         encoder_channels[d],
                                                         decoder_channels[d])

        self.convs = nn.ModuleDict(convs)

        self.final = nn.Sequential(
            nn.Conv2d(decoder_channels[0], 1, 3, padding=1),
            nn.Sigmoid())

    def forward(self, features):

        for d in range(self.depth - 2, -1, -1):
            features[d] = self.convs["conv{}".format(d)](features[d + 1], features[d])

        return self.final(features[0])


class Identity(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
    ):
        super().__init__()
        convs = []
        for ec, dc in zip(encoder_channels, decoder_channels):
            convs.append(nn.Conv2d(ec, dc, 1))
        self.convs = nn.ModuleList(convs)

    def forward(self, features):
        return [c(f) for f, c in zip(features, self.convs)]
