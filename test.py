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
            #

