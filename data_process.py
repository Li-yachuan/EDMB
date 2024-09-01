from torch.utils import data
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import random
import imageio
from pathlib import Path
from torch.nn.functional import interpolate
from skimage import morphology


class BSDS_VOCLoader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS_PASCAL', split='train',
                 transform=False, threshold=0.3, ablation=False):
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        print('Threshold for ground truth: %f on BSDS_VOC' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':

            self.filelist = join(root, "HED-BSDS/bsds_pascal_train_pair.lst")
        elif self.split == 'test':

            self.filelist = join(root, "HED-BSDS/test.lst")
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < threshold)] = 2
            lb[lb >= threshold] = 1

        else:
            img_file = self.filelist[index].rstrip()

        with open(join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class COD_Loader(data.Dataset):

    def __init__(self, root='/workspace/00Dataset/COD',
                 split='train', trainsize=500, cropsize=400, testsize=416):
        assert split in ["train", 'CAMO', 'CHAMELEON', 'COD10K', 'NC4K']
        self.root = root
        self.split = split
        self.cropsize = cropsize

        re_size = trainsize if self.split == "train" else testsize

        self.img_transform = transforms.Compose([
            transforms.Resize((re_size, re_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.ge_transform = transforms.Compose([
            transforms.Resize((re_size, re_size)),
            transforms.ToTensor()])

        self.filelist = join(self.root, '{}.lst'.format(self.split))

        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_file, lb_file, _ = self.filelist[index].split()
        img_file = img_file.strip()
        lb_file = lb_file.strip()
        assert isfile(join(self.root, img_file))
        assert isfile(join(self.root, lb_file))
        img = self.img_transform(Image.open(join(self.root, img_file)).convert("RGB"))
        lb = self.ge_transform(Image.open(join(self.root, lb_file)).convert("L"))

        return self.crop(img, lb, self.cropsize) \
            if self.split == "train" \
            else (img, lb, Path(img_file).stem)

    @staticmethod
    def crop(img, lb, crop_size):
        _, h, w = img.size()
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        img = img[:, i:i + crop_size, j:j + crop_size]
        lb = lb[:, i:i + crop_size, j:j + crop_size]

        return img, lb


class SOD_Loader(data.Dataset):

    def __init__(self, root='/workspace/00Dataset/SOD',
                 split='train', trainsize=500, cropsize=400,
                 # testsize=416
                 testsize=384):
        assert split in ["DUTS-TR", "DUTS-TE", "DUT-OMRON", "HKU-IS", "ECSSD", "PASCAL-S"]
        self.root = root
        self.split = split
        self.cropsize = cropsize

        re_size = trainsize if self.split == "DUTS-TR" else testsize

        self.img_transform = transforms.Compose([
            transforms.Resize((re_size, re_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.ge_transform = transforms.Compose([
            transforms.Resize((re_size, re_size)),
            transforms.ToTensor()])

        self.filelist = join(self.root, '{}.lst'.format(self.split))

        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_file, lb_file = self.filelist[index].split()
        img_file = img_file.strip()
        lb_file = lb_file.strip()

        assert isfile(join(self.root, img_file))
        assert isfile(join(self.root, lb_file))

        img = self.img_transform(Image.open(join(self.root, img_file)).convert("RGB"))
        lb = self.ge_transform(Image.open(join(self.root, lb_file)).convert("L"))

        return self.crop(img, lb, self.cropsize) \
            if self.split == "DUTS-TR" \
            else (img, lb, Path(img_file).stem)

    @staticmethod
    def crop(img, lb, crop_size):
        _, h, w = img.size()
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        img = img[:, i:i + crop_size, j:j + crop_size]
        lb = lb[:, i:i + crop_size, j:j + crop_size]

        return img, lb


class CHASE_Loader(data.Dataset):
    """
    Dataloader CHASE
    """

    def __init__(self, root='/workspace/00Dataset/CHASEDB1', split='train'):
        self.root = root
        self.split = split
        self.cropsize = 380
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        if self.split == 'train':
            self.filelist = join(self.root, 'train_pair.lst')
        else:
            self.filelist = join(self.root, 'test.lst')
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()
        self.filelist = [i for i in self.filelist if "s0.25" not in i]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_file, lb_file, _ = self.filelist[index].strip("\n").split(" ")
        lb = Image.open(join(self.root, self.split, lb_file)).convert("L")
        lb = transforms.ToTensor()(lb)
        img = Image.open(join(self.root, self.split, img_file)).convert('RGB')
        img = self.transform(img)
        return self.crop(img, lb, self.cropsize) \
            if self.split == "train" \
            else (img, lb, Path(img_file).stem)

    @staticmethod
    def crop(img, lb, crop_size):
        _, h, w = img.size()

        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        img = img[:, i:i + crop_size, j:j + crop_size]
        lb = lb[:, i:i + crop_size, j:j + crop_size]

        return img, lb


class BSDS_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS', split='train', threshold=0.3,
                 colorJitter=False, mix=False):
        self.root = root
        self.split = split
        self.threshold = threshold
        self.mix = mix
        print('Threshold for ground truth: %f on BSDS' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if colorJitter:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        if self.split == 'train':
            self.filelist = join(self.root, 'train_BSDS.lst')
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file = img_lb_file[0]

            if self.mix:
                label_list = [transforms.ToTensor()(Image.open(join(self.root, lb_file)))
                              for lb_file in img_lb_file[1:]]

                lb = torch.cat(label_list, 0).mean(0, keepdim=True)
                lb[lb >= self.threshold] = 1
                lb[(lb > 0) & (lb < self.threshold)] = 2

            else:
                lb_index = random.randint(2, len(img_lb_file)) - 1
                lb_file = img_lb_file[lb_index]
                lb = transforms.ToTensor()(Image.open(join(self.root, lb_file)))


        else:
            img_file = self.filelist[index].rstrip()

        with open(join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class PASCAL_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS'):
        self.root = root
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        self.filelist = join(self.root, 'train_PASCAL.lst')

        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_file, lb_file = self.filelist[index].strip("\n").split(" ")
        lb = Image.open(join(self.root, lb_file)).convert("L")
        lb = transforms.ToTensor()(lb)
        img = Image.open(join(self.root, img_file)).convert('RGB')
        img = self.transform(img)
        return img, lb


class NYUD_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', mode="RGB"):
        self.root = root
        self.split = split
        #
        if mode == "RGB":
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        else:  # calculate by Archer
            normalize = transforms.Normalize(mean=[0.519, 0.370, 0.465],
                                             std=[0.226, 0.246, 0.186])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        if self.split == 'train':
            if mode == "RGB":
                self.filelist = join(root, "image-train.lst")
            else:
                self.filelist = join(root, "hha-train.lst")

        elif self.split == 'test':
            if mode == "RGB":
                self.filelist = join(root, "image-test.lst")
            else:
                self.filelist = join(root, "hha-test.lst")

        else:
            raise ValueError("Invalid split type!")

        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].strip("\n").split(" ")

        else:
            img_file = self.filelist[index].strip("\n").split(" ")[0]

        img = imageio.imread(join(self.root, img_file))
        img = self.transform(img)

        if self.split == "train":

            label = transforms.ToTensor()(Image.open(join(self.root, lb_file)).convert('L'))
            label = (label - label.min()) / (label.max() - label.min())

            img, label = self.crop(img, label)

            label[label >= 0.5] = 1
            label[label < 0.2] = 0
            label[(label >= 0.2) & (label < 0.5)] = 2
            # # following TEED
            # label[label > 0.1] += 0.2  # 0.4
            # label = torch.clip(label, 0., 1.)

            return img, label

        else:
            img_name = Path(img_file).stem
            return img, img_name

    @staticmethod
    def crop(img, lb):
        _, h, w = img.size()
        crop_size = 400

        if h < crop_size or w < crop_size:
            resize_scale = round(max(crop_size / h, crop_size / w) + 0.1, 1)

            img = interpolate(img.unsqueeze(0), scale_factor=resize_scale, mode="bilinear").squeeze(0)
            lb = interpolate(lb.unsqueeze(0), scale_factor=resize_scale, mode="nearest").squeeze(0)
        _, h, w = img.size()
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        img = img[:, i:i + crop_size, j:j + crop_size]
        lb = lb[:, i:i + crop_size, j:j + crop_size]

        return img, lb


class BIPED_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root=' ', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(root, "train_pair.lst")

        elif self.split == 'test':
            self.filelist = join(root, "test.lst")
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()
        # print(self.filelist)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].strip("\n").split(" ")

        else:
            img_file = self.filelist[index].rstrip()

        img = imageio.imread(join(self.root, img_file))
        img = transforms.ToTensor()(img)

        if self.split == "train":
            label = transforms.ToTensor()(Image.open(join(self.root, lb_file)).convert('L'))
            img, label = self.crop(img, label)

            label[label >= 0.5] = 1
            label[label < 0.2] = 0
            label[(label >= 0.2) & (label < 0.5)] = 2

            return img, label

        else:
            img_name = Path(img_file).stem
            return img, img_name

    @staticmethod
    def crop(img, lb):
        _, h, w = img.size()
        assert (h > 400) and (w > 400)
        crop_size = 400
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        img = img[:, i:i + crop_size, j:j + crop_size]
        lb = lb[:, i:i + crop_size, j:j + crop_size]

        return img, lb


class Multicue_Loader(data.Dataset):
    """
    Dataloader for Multicue
    """

    def __init__(self, root='data/', split='train', transform=False, threshold=0.3, setting=['boundary', '1']):
        """
        setting[0] should be 'boundary' or 'edge'
        setting[1] should be '1' or '2' or '3'
        """
        self.root = root
        self.split = split
        self.threshold = threshold
        print('Threshold for ground truth: %f on setting %s' % (self.threshold, str(setting)))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            self.filelist = join(self.root, 'train_pair_%s_set_%s.lst' % (setting[0], setting[1]))
        elif self.split == 'test':
            self.filelist = join(self.root, 'test_%s_set_%s.lst' % (setting[0], setting[1]))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()

            lb = transforms.ToTensor()(Image.open(join(self.root, lb_file)).convert("L"))

            lb[lb > self.threshold] = 1
            lb[(lb > 0) & (lb < self.threshold)] = 2

        else:
            img_file = self.filelist[index].rstrip()

        with open(join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return self.crop(img, lb)
        else:
            img_name = Path(img_file).stem
            return img, img_name

    @staticmethod
    def crop(img, lb):
        _, h, w = img.size()
        crop_size = 400

        if (h < crop_size) or (w < crop_size):
            resize_scale = round(max(crop_size / h, crop_size / w) + 0.1, 1)

            img = interpolate(img.unsqueeze(0), scale_factor=resize_scale, mode="bilinear").squeeze(0)
            lb = interpolate(lb.unsqueeze(0), scale_factor=resize_scale, mode="nearest").squeeze(0)
        _, h, w = img.size()
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        img = img[:, i:i + crop_size, j:j + crop_size]
        lb = lb[:, i:i + crop_size, j:j + crop_size]

        return img, lb


from torch import nn
import torch.nn.functional as F


def NMS(img, r=2, m=1.01):
    img = img.unsqueeze(0)

    filter_size = 3
    generated_filters = torch.tensor([[0.25, 0.5, 0.25]])
    E = F.conv2d(input=img, weight=generated_filters.unsqueeze(0).unsqueeze(0), padding=(0, filter_size // 2))

    E = F.conv2d(input=E, weight=generated_filters.T.unsqueeze(0).unsqueeze(0), padding=(filter_size // 2, 0))

    filter_size = 9
    generated_filters = torch.tensor([[0.04, 0.08, 0.12, 0.16, 0.2, 0.16, 0.12, 0.08, 0.04]])
    E1 = F.conv2d(E, weight=generated_filters.unsqueeze(0).unsqueeze(0), padding=(0, filter_size // 2))
    E1 = F.conv2d(E1, weight=generated_filters.T.unsqueeze(0).unsqueeze(0), padding=(filter_size // 2, 0))

    sobel_filter = torch.tensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]]).float()

    grad_x = F.conv2d(E1, weight=sobel_filter.unsqueeze(0).unsqueeze(0), padding=sobel_filter.shape[0] // 2)
    grad_y = F.conv2d(E1, weight=sobel_filter.T.unsqueeze(0).unsqueeze(0), padding=sobel_filter.shape[0] // 2)

    grad_xy = F.conv2d(grad_x, weight=sobel_filter.T.unsqueeze(0).unsqueeze(0), padding=sobel_filter.shape[0] // 2)
    grad_xx = F.conv2d(grad_x, weight=sobel_filter.unsqueeze(0).unsqueeze(0), padding=sobel_filter.shape[0] // 2)
    grad_yy = F.conv2d(grad_y, weight=sobel_filter.T.unsqueeze(0).unsqueeze(0), padding=sobel_filter.shape[0] // 2)

    O = torch.atan(grad_yy * torch.sign(-grad_xy) / (grad_xx + 1e-5)) % 3.14

    ## edgesNmsMex(E, O, 2, 5, 1.01)
    E *= m
    coso = torch.cos(O)
    sino = torch.sin(O)
    _, _, H, W = img.size()
    norm = torch.tensor([[[[W, H]]]]).type_as(E)
    h = torch.linspace(-1.0, 1.0, H).view(-1, 1).repeat(1, W)
    w = torch.linspace(-1.0, 1.0, W).repeat(H, 1)
    grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2).type_as(E)
    E0 = E.clone()
    for d in range(-r, r + 1):
        if d == 0: continue
        grid1 = grid - torch.stack((coso * d, sino * d), dim=-1).squeeze() / norm
        neighber = F.grid_sample(E, grid1, align_corners=True)

        E0[neighber > E0] = 0
    E0[E0 > 0] = 1
    return E0.squeeze(0)
