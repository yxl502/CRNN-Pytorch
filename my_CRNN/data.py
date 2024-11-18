from captcha.image import ImageCaptcha, WheezyCaptcha
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
from config import char_list, batch_size


class CaptchaData(Dataset):
    def __init__(self, char_list, num=100):
        self.char_list = char_list
        self.char2index = {
            self.char_list[i]: i for i in range(len(self.char_list))
        }

    def __getitem__(self, item):
        chars = ''
        for i in range(np.random.randint(1, 10)):
            chars += self.char_list[np.random.randint(len(char_list))]

        image = ImageCaptcha(width=40*len(chars), height=60).generate_image(chars)
        # image = WheezyCaptcha(width=40*len(chars), height=60).generate_image(chars)
        chars_tensor = self._numerical(chars)
        # image_tensor = self._totensor(image)
        return image, chars_tensor

    def _numerical(self, chars):
        chars_tensor = torch.zeros(len(chars))
        for i in range(len(chars)):
            chars_tensor[i] = self.char2index[chars[i]] + 1
        return chars_tensor

    def _totensor(self, image):
        return transforms.ToTensor()(image)

    def __len__(self):
        return 10000


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.transform = transforms.Compose(
            [
                transforms.ColorJitter(),
                transforms.RandomRotation(degrees=(0, 5)),
                transforms.ToTensor()
            ]
        )

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.transform(img)
        img.sub_(0.5).div_(0.5)
        return img


class alignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images = [b[0].convert('L') for b in batch]
        labels = [b[1] for b in batch]

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratio = []
            for image in images:
                w, h = image.size
                ratio.append(w / float(h))
            ratio.sort()
            max_ratio = ratio[-1]
            imgW = int(np.floor(max_ratio*imgH))
            imgW = max(imgH * self.min_ratio, imgW)

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels


data = CaptchaData(char_list)
c = alignCollate()
train_dl = DataLoader(data, batch_size, collate_fn=c, num_workers=4)
test_dl = DataLoader(data, batch_size*2, collate_fn=c, num_workers=4)