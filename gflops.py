from model.basemodel import Basemodel
import argparse
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument("--encoder", default="MIXENC_PNG",
                    help="caformer-m36,Dul-M36,DUL-Mamba")
parser.add_argument("--decoder", default="MIXUNET",
                    help="unet,unetp,default")
parser.add_argument("--global_ckpt",
                    default="/workspace/EDMamba/output-VM/0602-bsds-s-mixlb/"
                            "epoch-0-training-record/epoch-0-checkpoint.pth")
args = parser.parse_args()


net = Basemodel(args).cuda()
input = torch.randn(1, 3, 481, 321)  # batchsize=1, 输入向量长度为10
# input = torch.randn(1, 3, 560, 425)  # batchsize=1, 输入向量长度为10
flops = FlopCountAnalysis(net, input.cuda())
print("FLOPs: ", flops.total() / 1000 / 1000 / 1000)



from thop import profile

input = torch.randn(1, 3, 481, 321)
flops, params = profile(net, inputs=(input.cuda(), ))
print("FLOPs=", str(flops/1e9) + '{}'.format("G"))