from torch import nn
import torch
from torch.nn import functional as F
from model.vmamba import VSSBlock
from model.unetp import Conv2dReLU


class DecoderBlock_vss(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
    ):
        super().__init__()
        self.vss1 = VSSBlock(
            hidden_dim=in_channels,
            drop_path=0.1,
            norm_layer=nn.LayerNorm,
            channel_first=False,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
        )
        self.vss2 = VSSBlock(
            hidden_dim=in_channels,
            drop_path=0.1,
            norm_layer=nn.LayerNorm,
            channel_first=False,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
        )
        self.pre_process = nn.Conv2d(in_channels + skip_channels,
                                     in_channels,
                                     kernel_size=1, padding=0)

        self.post_process = nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=1, padding=0)

    def forward(self, x, skip=None):
        x = F.interpolate(x, size=skip.size()[2:], mode="bilinear")
        x = torch.cat([x, skip], dim=1)
        x = self.pre_process(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x = self.vss1(x)
        x = self.vss2(x)
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.post_process(x)
        return x


class DecoderBlock_ir(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, size=skip.size()[2:], mode="bilinear")
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
    ):
        super().__init__()

        # encoder_channels: 16,32,96,192,384
        # decoder_channel:  32,64, 128,256

        self.width = len(encoder_channels)
        convs = dict()
        for w in range(self.width - 1):
            for d in range(self.width - w - 1):
                DecoderBlock = DecoderBlock_ir if d < 2 else DecoderBlock_vss
                # DecoderBlock = DecoderBlock_vss
                if w == 0:
                    convs["conv{}_{}".format(d, w)] = DecoderBlock(encoder_channels[d + 1],
                                                                   encoder_channels[d],
                                                                   decoder_channels[d])
                else:
                    convs["conv{}_{}".format(d, w)] = DecoderBlock(decoder_channels[d + 1],
                                                                   decoder_channels[d],
                                                                   decoder_channels[d])

        self.convs = nn.ModuleDict(convs)

        self.final = nn.Sequential(
            nn.Conv2d(decoder_channels[0], 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        for w in range(self.width - 1):
            for d in range(self.width - w - 1):
                features[d] = self.convs["conv{}_{}".format(d, w)](features[d + 1], features[d])
        return self.final(features[0])
        # return features
