import os
import numpy as np
import sys
from scipy.interpolate import interp1d
from utils_ois import computeRPF, computeRPF_numpy, findBestRPF
import shutil
from glob import glob

assert len(sys.argv) == 2

# exp_dir = '/data/zhoucaixia/workspace/UD_Edge/tmp/trainval_sigma_logit_unetpp_alpha_ffthalf_feat_testalpha_clipsum/alpha_style_all_epoch19/'
exp_dir = sys.argv[1]

model_name = os.listdir(exp_dir)
if "multiGranu" in model_name:
    model_name.pop(model_name.index("multiGranu"))
if "record.txt" in model_name:
    model_name.pop(model_name.index("record.txt"))
# assert len(model_name) == 11 or len(model_name) == 3

record_txt = open(os.path.join(exp_dir, "record.txt"), 'w')

model_name = ["-5.0","-4.5","-4.0","-3.5","-3.0","-2.5","-2.0","-1.5","-1.0","-0.5","0.0"]
# model_name = ["-3","-2.5","-2","-1.5","-1","0","3","2.5","2","1.5","1"]

# model_name = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# print(os.path.join(exp_dir, model_name[0]))
# exit()
if os.path.isdir(os.path.join(exp_dir, model_name[0], "nms-eval")):
    nms_dir = "nms-eval"
elif os.path.isdir(os.path.join(exp_dir, model_name[0], "nms-eval-lite")):
    nms_dir = "nms-eval-lite"
else:
    nms_dir = None
    raise Exception("NO Avaliable NMS Data")
eval_list = glob(os.path.join(exp_dir, model_name[0], nms_dir, "*_ev1.txt"))

best_ods_dir = os.path.join(exp_dir, "multiGranu/nms-eval")
best_ois_png = os.path.join(exp_dir, "multiGranu/best_ois/png")
best_ois_mat = os.path.join(exp_dir, "multiGranu/best_ois/mat")
os.makedirs(best_ods_dir, exist_ok=True)
os.makedirs(best_ois_png, exist_ok=True)
os.makedirs(best_ois_mat, exist_ok=True)

eps = sys.float_info.epsilon
np.seterr(divide='ignore', invalid='ignore')
Z = np.zeros((99, 1))

cntR_all, sumR_all, cntP_all, sumP_all = Z, Z, Z, Z
T = np.array([i / 100 for i in range(1, 100)])
# [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
oisCntR, oisSumR, oisCntP, oisSumP = 0, 0, 0, 0


def index_select(arr1, index_arr):
    '''
    
    :param arr1: model_F (11,9)
    :param index_arr: index_selected (9,)
    :return: 
    '''

    return_arr = np.zeros((99, 1))
    for i in range(len(index_arr)):
        return_arr[i][0] = arr1[index_arr[i]][i]

    return return_arr


for eval_name in eval_list:

    eval = eval_name.split("/")[-1]
    new_file_name = os.path.join(best_ods_dir, eval)
    new_file = open(new_file_name, "w")
    model_cntR, model_sumR, model_cntP, model_sumP, model_R, model_P, model_F = [], [], [], [], [], [], []
    model_count = 0
    for model in model_name:
        eval_path = os.path.join(exp_dir, str(model), nms_dir, eval)
        eval_txt = open(eval_path, "r").readlines()
        for index in range(len(eval_txt)):
            str_lines = ' '.join(eval_txt[index].split())
            str_lines = str_lines.strip("\n").split(" ")
            cntR, sumR, cntP, sumP = \
                int(str_lines[1]), \
                int(str_lines[2]), \
                int(str_lines[3]), \
                int(str_lines[4])
            model_cntR.append(cntR)
            model_sumR.append(sumR)
            model_cntP.append(cntP)
            model_sumP.append(sumP)

            R, P, F = computeRPF(cntR, sumR, cntP, sumP)
            model_R.append(R)
            model_P.append(P)
            model_F.append(F)

    model_R = np.array(model_R).reshape(len(model_name), 99)
    model_p = np.array(model_P).reshape(len(model_name), 99)
    model_F = np.array(model_F).reshape(len(model_name), 99)

    index_selected = np.argmax(model_F, 0)  # 取值范围0-10,各个阈值下选择哪个模型

    img_F = index_select(model_F, index_selected)

    max_F_index = np.argmax(img_F)  # 选择对于该张图片来说最好的阈值

    i_max_loc = np.where(model_F == np.max(model_F))[0][0]

    model_cntR, model_sumR, model_cntP, model_sumP = \
        np.array(model_cntR).reshape(len(model_name), 99), \
        np.array(model_sumR).reshape(len(model_name), 99), \
        np.array(model_cntP).reshape(len(model_name), 99), \
        np.array(model_sumP).reshape(len(model_name), 99)

    i_eval = eval.rindex("_")
    record_txt.writelines(eval[:i_eval] + "\t" + str(i_max_loc) + "\t" + model_name[i_max_loc] + "\n")

    img_nm = eval.replace("_ev1.txt", "")
    mat_source = os.path.join(exp_dir, model_name[i_max_loc], "mat", "{}.mat".format(img_nm))
    mat_dest = os.path.join(best_ois_mat, "{}.mat".format(img_nm))
    png_source = os.path.join(exp_dir, model_name[i_max_loc], "png", "{}.png".format(img_nm))
    png_dest = os.path.join(best_ois_png, "{}.png".format(img_nm))
    shutil.copy(mat_source, mat_dest)
    shutil.copy(png_source, png_dest)

    index_model = index_selected[max_F_index]
    # print(index_model)
    oisCntR, oisSumR, oisCntP, oisSumP = \
        oisCntR + model_cntR[index_model][max_F_index], \
        oisSumR + model_sumR[index_model][max_F_index], \
        oisCntP + model_cntP[index_model][max_F_index], \
        oisSumP + model_sumP[index_model][max_F_index]

    img_cntR, img_sumR, img_cntP, img_sumP = \
        index_select(model_cntR, index_selected), \
        index_select(model_sumR, index_selected), \
        index_select(model_cntP, index_selected), \
        index_select(model_sumP, index_selected)

    cntR_all, sumR_all, cntP_all, sumP_all = \
        cntR_all + img_cntR, \
        sumR_all + img_sumR, \
        cntP_all + img_cntP, \
        sumP_all + img_sumP

    th = np.arange(0.01, 1, 0.01)
    for i_th in range(99):
        new_file.writelines(
            "\t\t" + ('%.2f' % th[i_th]) + '      '
            + str(int(img_cntR[i_th].item())) + "      "
            + str(int(img_sumR[i_th].item())) + "      "
            + str(int(img_cntP[i_th].item())) + "      "
            + str(int(img_sumP[i_th].item())) + "\n")
    new_file.flush()
    new_file.close()

R_all, P_all, F_all = computeRPF_numpy(cntR_all, sumR_all, cntP_all, sumP_all)

odsR, odsP, odsF, odsT = findBestRPF(T, R_all, P_all)
oisR, oisP, oisF = computeRPF(oisCntR, oisSumR, oisCntP, oisSumP)

# AP
new_R = np.arange(0.01, 1.01, 0.01)
f = interp1d(R_all[:, 0], P_all[:, 0], kind='linear', bounds_error=False)
bestAP = np.nansum(f(new_R)) / 100

info = 'best ODS:' + str(odsF) + ', best OIS:' + str(oisF) + ", best AP:" + str(bestAP)
print(info)
record_txt.writelines(info)
record_txt.flush()
record_txt.close()
