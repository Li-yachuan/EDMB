from torch import nn
import torch
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)

    def forward(self, x):
        return self.attention(x)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.InstanceNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):

        if skip is not None:
            x = F.interpolate(x, size=skip.size()[2:], mode="bilinear")
            # x = F.interpolate(x, size=skip.size()[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            # x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels
    ):
        super().__init__()

        self.depth = len(encoder_channels)
        convs = dict()

        for d in range(self.depth - 1):
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

        return self.final(features[0]), features[0]


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


class MixDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            ckpt="output-VM/0613-bsds-su/epoch-11-training-record/epoch-11-checkpoint.pth"
    ):
        super().__init__()

        self.decoder = UnetDecoder(encoder_channels, decoder_channels)
        self.load_pretrained(ckpt)
        self.decoder.eval()
        for k, v in self.decoder.named_parameters():
            v.requires_grad = False

        self.local_decoder = UnetDecoder(encoder_channels, decoder_channels)
        self.local_decoder2 = UnetDecoder(encoder_channels, decoder_channels)

        self.fuse_head = Local8x8_fuse_head(decoder_channels[0], decoder_channels[0],activate=nn.Identity)
        self.fuse_head2 = Local8x8_fuse_head(decoder_channels[0], decoder_channels[0],activate=nn.Softplus)

    def load_pretrained(self, ckpt=None, key="state_dict"):
        if ckpt is not None:
            try:
                _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
                incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
                # print(incompatibleKeys)
            except Exception as e:
                print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, feature):
        global_feat, local_feat_ = feature

        # for i in global_feat:
        #     print(i.size())
        # print(("****************"))
        # for j in local_feat_:
        #     print(j.size())
        # exit()
        global_edge, global_feat = self.decoder(global_feat)

        local_edge, local_feat = self.local_decoder(local_feat_[:])
        edge = self.fuse_head(local_feat, global_feat)

        local_edge2, local_feat2 = self.local_decoder2(local_feat_)
        edge2 = self.fuse_head2(local_feat2, global_feat)

        return [edge, edge2, local_edge, local_edge2, global_edge]


###########################################################################
# Created by: pmy
# Copyright (c) 2019
##########################################################################


class SFTLayer(nn.Module):
    def __init__(self, head_channels):
        super(SFTLayer, self).__init__()

        self.SFT_scale_conv0 = nn.Conv2d(head_channels, head_channels, 1)
        self.SFT_scale_conv1 = nn.Conv2d(head_channels, head_channels, 1)

        self.SFT_shift_conv0 = nn.Conv2d(head_channels, head_channels, 1)
        self.SFT_shift_conv1 = nn.Conv2d(head_channels, head_channels, 1)

    def forward(self, local_features, global_features):
        scale = self.SFT_scale_conv1(F.relu(self.SFT_scale_conv0(global_features), inplace=True))
        shift = self.SFT_shift_conv1(F.relu(self.SFT_shift_conv0(global_features), inplace=True))
        fuse_features = local_features * (scale + 1) + shift
        return fuse_features


class Local8x8_fuse_head(nn.Module):
    def __init__(self,
                 mla_channels=128,
                 mlahead_channels=128,
                 norm_layer=nn.BatchNorm2d,
                 activate=nn.Identity):
        super(Local8x8_fuse_head, self).__init__()

        self.channels = mla_channels
        self.head_channels = mlahead_channels
        self.BatchNorm = norm_layer
        self.activate = activate()

        self.SFT_head = SFTLayer(self.head_channels)
        self.edge_head = nn.Sequential(
            nn.Conv2d(self.head_channels, self.head_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.head_channels), nn.ReLU(),
            nn.Conv2d(self.head_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, local_features, global_features):

        fuse_features = self.SFT_head(local_features, global_features)
        fuse_edge = self.edge_head(fuse_features)

        fuse_edge = self.activate(fuse_edge)
        return fuse_edge
