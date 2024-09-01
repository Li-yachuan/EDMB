from os.path import join
import os
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
import scipy.io as sio
from scipy import stats
from torch.distributions import Normal, Independent


def test(model, test_loader, save_dir, mg=False):
    print("single scale test")

    model.eval()
    for idx, (image, filename) in enumerate(tqdm(test_loader)):
        image = image.cuda()
        if not mg:
            with torch.no_grad():
                outputs = model(image)

            if isinstance(outputs, list) and len(outputs) == 5:
                mean, std, _, _, _ = outputs
                outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)

                outputs = [outputs_dist.rsample() for _ in range(100)]
                outputs = torch.cat(outputs, dim=1).mean(dim=1, keepdim=True)
                outputs = torch.sigmoid(outputs)

                result = torch.squeeze(outputs.detach()).cpu().numpy()
                result = np.clip(result, stats.mode(result.reshape(-1), keepdims=False).mode.item(), 1)

            elif isinstance(outputs, list) and len(outputs) == 3:
                result = torch.squeeze(outputs[0].detach()).cpu().numpy()
            elif isinstance(outputs, torch.Tensor):
                result = torch.squeeze(outputs.detach()).cpu().numpy()
            else:
                raise Exception("not avaliable")

            result = (result - result.min()) / (result.max() - result.min())

            result_png = Image.fromarray((result * 255).astype(np.uint8))

            png_save_dir = os.path.join(save_dir, "png")
            mat_save_dir = os.path.join(save_dir, "mat")
            if not os.path.exists(png_save_dir):
                os.makedirs(png_save_dir)
            if not os.path.exists(mat_save_dir):
                os.makedirs(mat_save_dir)

            result_png.save(join(png_save_dir, "%s.png" % filename))
            sio.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)

        else:
            # muge=[-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3]
            # muge = [-4, -3.5, -3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2]
            # muge = [i / 2 for i in range(-11, 1)]
            muge = [-5,-4.5,-4,-3.5,-0.5]
            for granu in muge:
                with torch.no_grad():
                    outputs = model(image)
                if isinstance(outputs, list):
                    mean, std, _, _, _ = outputs

                outputs = torch.sigmoid(mean + std * granu)
                outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
                result = torch.squeeze(outputs.detach()).cpu().numpy()

                result = np.clip(result, stats.mode(result.reshape(-1), keepdims=False).mode.item(), 1)
                result = (result - result.min()) / (result.max() - result.min())

                result_png = Image.fromarray((result * 255).astype(np.uint8))

                png_save_dir = os.path.join(save_dir, str(granu), "png")
                mat_save_dir = os.path.join(save_dir, str(granu), "mat")

                if not os.path.exists(png_save_dir):
                    os.makedirs(png_save_dir)

                if not os.path.exists(mat_save_dir):
                    os.makedirs(mat_save_dir)
                result_png.save(join(png_save_dir, "%s.png" % filename))
                sio.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)

            # with torch.no_grad():
            #     for l in alpha_list:
            #         label_bias = torch.ones(1).cuda()
            #         label_bias = label_bias * l
            #         outputs = model(image, label_bias)
            #         if isinstance(outputs, list):
            #             outputs = outputs[0]
            #
            #         outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
            #         result = torch.squeeze(outputs.detach()).cpu().numpy()
            #         result_png = Image.fromarray((result * 255).astype(np.uint8))
            #         if len(alpha_list)>1:
            #             png_save_dir = os.path.join(save_dir, str(l), "png")
            #             mat_save_dir = os.path.join(save_dir, str(l), "mat")
            #         else:
            #             png_save_dir = os.path.join(save_dir, "png")
            #             mat_save_dir = os.path.join(save_dir, "mat")
            #         if not os.path.exists(png_save_dir):
            #             os.makedirs(png_save_dir)
            #         if not os.path.exists(mat_save_dir):
            #             os.makedirs(mat_save_dir)
            #
            #         result_png.save(join(png_save_dir, "%s.png" % (filename)))
            #         sio.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)
            #


import cv2


def multiscale_test(model, test_loader, save_dir, scale_num=7, mg=False):
    png_save_dir = os.path.join(save_dir, "png")
    mat_save_dir = os.path.join(save_dir, "mat")
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    if not os.path.exists(mat_save_dir):
        os.makedirs(mat_save_dir)

    model.eval()
    if scale_num == 7:
        print("7 scale test")
        scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    else:
        print("3 scale test")
        scale = [0.5, 1.0, 1.5]

    for idx, (image, filename) in enumerate(tqdm(test_loader)):
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)
        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = torch.from_numpy(im_.transpose((2, 0, 1))).unsqueeze(0)
            with torch.no_grad():
                if not mg:
                    outputs = model(im_.cuda())
                    if isinstance(outputs, list):
                        outputs = outputs[0]
                # else:
                #     ## 11 is the best result in MuGE
                #     outputs, _, _ = model(im_.cuda(),11)

            result = torch.squeeze(outputs.detach()).cpu().numpy()

            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse
        multi_fuse = np.clip(multi_fuse / len(scale), 0, 1)

        result_png = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result_png.save(join(png_save_dir, "%s.png" % filename))

        sio.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': multi_fuse}, do_compression=True)


from functools import partial


def __identity(x):
    return x


def enhence_test(model, test_loader, save_dir):
    print("rotate enhence test")
    png_save_dir = os.path.join(save_dir, "png")
    mat_save_dir = os.path.join(save_dir, "mat")
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    if not os.path.exists(mat_save_dir):
        os.makedirs(mat_save_dir)

    model.eval()
    funcs = [partial(__identity),
             partial(cv2.rotate, rotateCode=cv2.ROTATE_90_CLOCKWISE),
             partial(cv2.rotate, rotateCode=cv2.ROTATE_180),
             partial(cv2.rotate, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)]

    funcs_t = [partial(__identity),
               partial(cv2.rotate, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE),
               partial(cv2.rotate, rotateCode=cv2.ROTATE_180),
               partial(cv2.rotate, rotateCode=cv2.ROTATE_90_CLOCKWISE)]

    for idx, (image, filename) in enumerate(tqdm(test_loader)):

        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))

        H, W, _ = image_in.shape

        multi_fuse = np.zeros((H, W), np.float32)

        for func, funct in zip(funcs, funcs_t):
            img = func(image_in)
            edge = __enhence_test_single(img, model)
            edge = funct(edge)
            multi_fuse += edge

        image_inf = cv2.flip(image_in, 1)  # shuiping fanzhuan
        multi_fuse_f = np.zeros((H, W), np.float32)

        for func, funct in zip(funcs, funcs_t):
            img = func(image_inf)
            edge = __enhence_test_single(img, model)
            edge = funct(edge)
            multi_fuse_f += edge

        multi_fuse = multi_fuse + cv2.flip(multi_fuse_f, 1)

        multi_fuse = multi_fuse / 8

        result_png = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result_png.save(join(png_save_dir, "%s.png" % filename))

        sio.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': multi_fuse}, do_compression=True)


def bright_enhence_test(model, test_loader, save_dir):
    print("bright enhence test")
    png_save_dir = os.path.join(save_dir, "png")
    mat_save_dir = os.path.join(save_dir, "mat")
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    if not os.path.exists(mat_save_dir):
        os.makedirs(mat_save_dir)

    model.eval()
    for idx, (image, filename) in enumerate(tqdm(test_loader)):
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))

        H, W, _ = image_in.shape

        multi_fuse = np.zeros((H, W), np.float32)
        bright_intals = [(0, 0.5), (0.25, 0.75), (0.5, 1)]
        for internl in bright_intals:
            img = __bright_func(image_in, internl)
            edge = __enhence_test_single(img, model)
            multi_fuse += edge

        multi_fuse = multi_fuse / len(bright_intals)

        result_png = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result_png.save(join(png_save_dir, "%s.png" % filename))

        sio.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': multi_fuse}, do_compression=True)


from utils import Metrics


def test_vessel(model, test_loader, save_dir):
    png_save_dir = os.path.join(save_dir, "png")
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    metrics = Metrics()
    model.eval()
    for idx, (image, lb, filename) in enumerate(tqdm(test_loader)):
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        result = __enhence_test_single(image_in, model,
                                       scale=[0.5, 0.75, 1.0, 1.25])
        metrics(result, lb)
        result_png = Image.fromarray((result * 255).astype(np.uint8))
        result_png.save(join(png_save_dir, "%s.png" % filename))
    metrics.show()


def __bright_func(image_in, internl):
    threshold_min = image_in.min() + (image_in.max() - image_in.min()) * internl[0]
    threshold_max = image_in.min() + (image_in.max() - image_in.min()) * internl[1]

    enh_image = np.clip(image_in, threshold_min, threshold_max)

    scale_factor = (image_in.max() - image_in.min()) / (threshold_max - threshold_min)
    offset = image_in.min() - scale_factor * threshold_min
    enh_image = scale_factor * enh_image + offset

    image = (image_in + enh_image) / 2
    return image


def __enhence_test_single(image_in, model, scale=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
    H, W, _ = image_in.shape
    multi_fuse = np.zeros((H, W), np.float32)

    for k in range(0, len(scale)):
        im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
        im_ = torch.from_numpy(im_.transpose((2, 0, 1))).unsqueeze(0)
        with torch.no_grad():
            result = model(im_.cuda()).squeeze().cpu().numpy()
        fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
        multi_fuse += fuse
    multi_fuse = multi_fuse / len(scale)
    return multi_fuse


from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure


def test_cod(test_loader, model, testsvdir):
    # testsets = ['CAMO', 'CHAMELEON']
    # testsets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']

    for dataset in test_loader.keys():
        eval_info = '\nBegin to eval...\nImg generated in {}\n'.format(
            os.path.join(testsvdir, dataset))
        print(eval_info)
        os.makedirs(os.path.join(testsvdir, dataset), exist_ok=True)
        FM = Fmeasure()
        WFM = WeightedFmeasure()
        SM = Smeasure()
        EM = Emeasure()
        M = MAE()
        for idx, (image, gt, img_name) in enumerate(tqdm(test_loader[dataset])):
            with torch.no_grad():
                image = image.cuda()
                pred = model(image)[-1].squeeze().cpu().numpy() * 255
                gt = gt.squeeze().numpy() * 255
            FM.step(pred=pred, gt=gt)
            WFM.step(pred=pred, gt=gt)
            SM.step(pred=pred, gt=gt)
            EM.step(pred=pred, gt=gt)
            M.step(pred=pred, gt=gt)

            pred = Image.fromarray(pred.astype(np.uint8))
            pred.save(os.path.join(testsvdir, dataset, "%s.png" % img_name))

        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = M.get_results()["mae"]

        results = {
            "Smeasure": sm,  # structure-measure
            "wFmeasure": wfm,  # weighted F-measure
            "MAE": mae,  # mean absolute error
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),  # mean E-measure
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
        }

        eval_info = "SM: {}\n" \
                    "MEM: {}\n" \
                    "WFM: {}\n" \
                    "MAE: {}\n" \
            .format(sm, em["curve"].mean(), wfm, mae)
        # structure-measure
        # mean E-measure
        # weighted F-measure
        # mean absolute error
        eval_info += "#" * 50
        print(eval_info)
