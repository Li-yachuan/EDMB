from model.basemodel import Basemodel
import yaml
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch
import sys
from utils import get_model_parm_nums

cfg = r"config/semi-supervise/S18_MG.yaml"
with open(cfg, 'r', encoding='utf-8') as file:
    cfg = yaml.safe_load(file)
# import pdb; pdb.set_trace()
net = Basemodel(encoder_name="Dul-S18",
                # decoder_name="ATTMIX",  #UNETMIX  UNETPPMIX
                # decoder_name="UNETMIX",  #UNETMIX  UNETPPMIX
                decoder_name="UNETPPMIX",  #UNETMIX  UNETPPMIX
                head_name="Mixhead",
                cfg=cfg).cuda()

print("MODEL SIZE: {}".format(get_model_parm_nums(net)))

input = torch.randn(1, 3, 480, 320)  # batchsize=1

flops = FlopCountAnalysis(net, input.cuda())
print("FLOPs: ", flops.total() / 1000 / 1000 / 1000)



from thop import profile

input = torch.randn(1, 3, 480, 320)
flops, params = profile(net, inputs=(input.cuda(), ))
print("FLOPs=", str(flops/1e9) + '{}'.format("G"))