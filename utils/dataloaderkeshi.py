import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pylab as plt
import random
import numpy as np

class Datases_loader(Dataset):
    def __init__(self, root_images, root_masks, h, w, save_dir=None):
        super().__init__()
        self.root_images = root_images
        self.root_masks = root_masks
        self.h = h
        self.w = w
        self.images = []
        self.labels = []
        self.save_dir = r'DeepCrack_rgb'

        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        files = sorted(os.listdir(self.root_images))
        sfiles = sorted(os.listdir(self.root_masks))
        for i in range(len(sfiles)):
            img_file = os.path.join(self.root_images, files[i])
            mask_file = os.path.join(self.root_masks, sfiles[i])
            self.images.append(img_file)
            self.labels.append(mask_file)

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image = self.images[idx]
            mask = self.labels[idx]
        else:
            image = self.images[idx]
            mask = self.labels[idx]

        image = Image.open(image)
        mask = Image.open(mask)

        # 定义图像增强操作
        tf = transforms.Compose([
            transforms.Resize((int(self.h * 1.25), int(self.w * 1.25))),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(16, fill=144),
            transforms.CenterCrop((self.h, self.w)),
            transforms.ToTensor()
        ])

        image = image.convert('RGB')
        norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        seed = np.random.randint(131254)

        # 设置随机种子
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # 图像增强
        img = tf(image)

        # 保存增强后的图像
        if self.save_dir:
            img_to_save = transforms.ToPILImage()(img)  # 将张量转换为 PIL 图像
            save_path = os.path.join(self.save_dir, f'augmented_image_{idx}.png')
            img_to_save.save(save_path)

        # 标准化
        img = norm(img)

        # 设置随机种子
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # 掩膜处理
        mask = tf(mask)
        mask[mask > 0] = 1.  # 二值化掩膜

        sample = {'image': img, 'mask': mask}

        return sample
