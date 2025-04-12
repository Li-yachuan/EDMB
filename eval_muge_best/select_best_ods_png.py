import os
import numpy as np
from shutil import copy
from glob import glob
from utils_ois import computeRPF, computeRPF_numpy, findBestRPF
import sys

assert len(sys.argv) == 2
select_th = 0.11
# src_dir = '/data/zhoucaixia/workspace/UD_Edge/tmp/trainval_sigma_logit_unetpp_alpha_ffthalf_feat_testalpha_clipsum/alpha_style_all_epoch19//'
# src_dir = '/workspace/EDMamba/output-VM/0715-BSDS-stageII-test/epoch-11-checkpoint-ss'
src_dir = sys.argv[1]
dst_dir = os.path.join(src_dir + 'best_ods_pic/')

# model_name = ["0", '0.1', '0.2', "0.3", '0.4', '0.5', "0.6", '0.7', '0.8', '0.9', '1']
model_name = os.listdir(src_dir)
if "multiGranu" in model_name:
    model_name.pop(model_name.index("multiGranu"))
if "record.txt" in model_name:
    model_name.pop(model_name.index("record.txt"))
assert len(model_name) == 11 or len(model_name) == 3

eval_list = glob(os.path.join(src_dir, model_name[0], "nms-eval/*_ev1.txt"))
model_name_dict = {}
for i in range(len(model_name)):
    model_name_dict[i] = model_name[i]

for eval_name in eval_list:
    eval = eval_name.split("/")[-1]
    model_cntR, model_sumR, model_cntP, model_sumP, model_R, model_P, model_F = [], [], [], [], [], [], []

    for model in model_name:
        eval_path = os.path.join(src_dir, str(model), "nms-eval", eval)

        eval_txt = open(eval_path, "r").readlines()

        str_lines = ' '.join(eval_txt[int(select_th * 100)].split())

        str_lines = str_lines.strip("\n").split(" ")
        cntR, sumR, cntP, sumP = int(str_lines[1]), int(str_lines[2]), int(str_lines[3]), int(str_lines[4])
        model_cntR.append(cntR), model_sumR.append(sumR), model_cntP.append(cntP), model_sumP.append(sumP)

        R, P, F = computeRPF(cntR, sumR, cntP, sumP)
        model_R.append(R), model_P.append(P), model_F.append(F)

    model_R, model_p, model_F = \
        np.array(model_R).reshape(len(model_name)), \
        np.array(model_P).reshape(len(model_name)), \
        np.array(model_F).reshape(len(model_name))
    kind = np.where(model_F == np.max(model_F))[0].item()

    # kind=int(kind)/(len(model_name)-1)
    # if kind==0 or kind==1:
    #     kind=int(kind)
    name = eval[:eval.rindex("_")]
    src_png_name = os.path.join(src_dir, model_name_dict[kind], "png", name + ".png")
    src_mat_name = os.path.join(src_dir, model_name_dict[kind], "mat", name + ".mat")

    sv_png = os.path.join(dst_dir, "png")
    sv_mat = os.path.join(dst_dir, "mat")

    os.makedirs(sv_png, exist_ok=True)
    os.makedirs(sv_mat, exist_ok=True)

    dsc_png_name = os.path.join(sv_png, name + ".png")
    dsc_mat_name = os.path.join(sv_mat, name + ".mat")

    copy(src_png_name, dsc_png_name)
    copy(src_mat_name, dsc_mat_name)

    print(name)
    # print(src_file_name, dsc_file_name)
