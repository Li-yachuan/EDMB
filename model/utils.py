from torch import nn


def get_encoder(nm, Dulbrn=16,global_ckpt=None):
    if "CAFORMER-M36" == nm.upper():
        from model.caformer import caformer_m36_384_in21ft1k
        encoder = caformer_m36_384_in21ft1k(pretrained=True)
    elif "DUL-M36" == nm.upper():
        from model.caformer import caformer_m36_384_in21ft1k
        encoder = caformer_m36_384_in21ft1k(pretrained=True, Dulbrn=Dulbrn)
    elif "DUL-MAMBA-B" == nm.upper():
        from model.vmamba import Backbone_VSSM
        encoder = Backbone_VSSM(
            pretrained="model/vssm_base_0229_ckpt_epoch_237.pth",
            out_indices=(0, 1, 2),
            # out_indices=(0, 1, 2, 3),
            dims=128,
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
            # drop_path_rate=0.3,
            Dulbrn=Dulbrn)

    elif "DUL-MAMBA-T" == nm.upper():
        from model.vmamba import Backbone_VSSM
        encoder = Backbone_VSSM(
            pretrained="model/vssm_tiny_0230_ckpt_epoch_262.pth",
            out_indices=(0, 1, 2),
            # out_indices=(0, 1, 2, 3),
            dims=96,
            depths=(2, 2, 5, 0),
            ssm_d_state=1,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            ssm_conv=3,
            ssm_conv_bias=False,
            forward_type="v05_noz",  # v3_noz,
            mlp_ratio=4.0,
            downsample_version="v3",
            patchembed_version="v2",
            Dulbrn=Dulbrn)

    elif "DUL-MAMBA-S" == nm.upper():
        from model.vmamba import Backbone_VSSM
        encoder = Backbone_VSSM(
            pretrained="model/vssm_small_0229_ckpt_epoch_222.pth",
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

    elif "MAMBA-S" == nm.upper():
        from model.vmamba import Backbone_VSSM
        encoder = Backbone_VSSM(
            pretrained="model/vssm_small_0229_ckpt_epoch_222.pth",
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
            Dulbrn=None)

    elif "MIXENC" == nm.upper():
        from model.MixEncoder import MIXENC
        encoder = MIXENC(Dulbrn=Dulbrn)
    elif "MIXENC_PNG" == nm.upper():#partly no grad
        assert global_ckpt is not None
        from model.MixEncoder_PNG import MIXENC
        encoder = MIXENC(Dulbrn=Dulbrn,ckpt=global_ckpt)
    elif "MIXENC_SHARED" == nm.upper():#partly no grad
        assert global_ckpt is not None
        from model.MixEncoder_shared import MIXENC
        encoder = MIXENC(Dulbrn=Dulbrn,ckpt=global_ckpt)
    elif "DUL-S18" == nm.upper():
        from model.caformer import caformer_s18_384_in21ft1k
        encoder = caformer_s18_384_in21ft1k(pretrained=True, Dulbrn=Dulbrn)
    elif "VGG-16" == nm.upper():
        from model.vgg import VGG16_C
        encoder = VGG16_C(pretrain="model/vgg16.pth")
    elif "LCAL" == nm.upper():
        from model.localextro import LCAL
        encoder = LCAL(Dulbrn=Dulbrn)
    else:
        raise Exception("Error encoder")
    return encoder


def get_head(nm, channels):
    from model.detect_head import CoFusion_head, CSAM_head, CDCM_head, Default_head, Fusion_head
    if nm == "aspp":
        head = CDCM_head(channels)
    elif nm == "atten":
        head = CSAM_head(channels)
    elif nm == "cofusion":
        head = CoFusion_head(channels)
    elif nm == "fusion":
        head = Fusion_head(channels)
    elif nm == "default":
        head = Default_head(channels)
    else:
        raise Exception("Error head")
    return head


def get_decoder(nm, incs, oucs=None):
    if oucs is None:
        oucs = (32, 32, 64, 128, 384)

    if nm.upper() == "UNETP":
        from model.unetp import UnetDecoder
        decoder = UnetDecoder(incs, oucs[-len(incs):])
    elif nm.upper() == "UNET":
        from model.unet import UnetDecoder
        decoder = UnetDecoder(incs, oucs[-len(incs):])
    elif nm.upper() == "UNETCA":
        from model.MixCA import UnetDecoder
        decoder = UnetDecoder(incs, oucs[-len(incs):],edge_only=True)
    elif nm.upper() == "UNETPM":
        from model.unetp_mamba import UnetDecoder
        decoder = UnetDecoder(incs, oucs[-len(incs):])
    elif nm.upper() == "UNETM":
        from model.unetm import UnetDecoder
        decoder = UnetDecoder(incs, oucs[-len(incs):])
    elif nm.upper() == "MIXUNET":
        from model.Mixunet import MixDecoder
        decoder = MixDecoder(incs, oucs[-len(incs):])
    elif nm.upper() == "MIXCA":
        from model.MixCA import MixDecoder
        decoder = MixDecoder(incs, oucs[-len(incs):])
    elif nm.upper() == "DEFAULT":
        from model.unet import Identity
        decoder = Identity(incs, oucs[-len(incs):])
    elif nm.upper() == "NOGLOBAL":#"BIUNET"
        from model.biunet import MixDecoder
        decoder = MixDecoder(incs, oucs[-len(incs):])
    elif nm.upper() == "SINGLEGRAN":
        from model.Mixunet_single_gran import MixDecoder
        decoder = MixDecoder(incs, oucs[-len(incs):])

    else:
        raise Exception("Error decoder")
    return decoder


def get_enhence(nm, chs):

    if nm is None:
        enc = nn.ModuleList([nn.Identity() for i in chs])
    else:
        from model.enhc_block import MultipleGranularity
        enc = MultipleGranularity()
    return enc
