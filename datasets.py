import glob
import random
import os

import cv2
import numpy as np

from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, mode="train", transforms=None):
        super().__init__()
        self.transforms = transform.Compose(transforms)
        self.mode = mode
        if self.mode == 'train':
            self.files = sorted(glob.glob(os.path.join(root, "imgs") + "/*.*"))
            self.labels = sorted(glob.glob(os.path.join(root, "labels") + "/*.*"))
        else:
            self.labels = sorted(glob.glob(root + "/*.*"))
        self.set_attrs(total_len=len(self.labels))
        print(f"from {mode} split load {self.total_len} images.")

    def __getitem__(self, index):
        label_path = self.labels[index % len(self.labels)]
        photo_id = label_path.split('/')[-1][:-4]
        img_B = Image.open(label_path)
        img_B = Image.fromarray(np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3, 2))

        if self.mode == "train":
            img_A = Image.open(self.files[index % len(self.files)])
            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
            img_A = self.transforms(img_A)
        else:
            img_A = np.empty([1])
        img_B = self.transforms(img_B)

        return img_A, img_B, photo_id


if __name__ == '__main__':
    data_train_path = "/data/comptition/jittor/data/train"
    data_val_path = "/data/comptition/jittor/data/val_A_labels_cleaned"
    # transforms = [
    #     transform.Resize(size=(224, 224), mode=Image.BICUBIC),
    #     transform.ToTensor(),
    #     transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ]
    #
    # dataloader = ImageDataset(data_train_path, mode="train", transforms=None).set_attrs(
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=1,
    # )
    # val_dataloader = ImageDataset(data_val_path, mode="val", transforms=None).set_attrs(
    #     batch_size=10,
    #     shuffle=False,
    #     num_workers=1,
    # )
    import matplotlib.pyplot as plt
    import tqdm

    conut = []
    files = glob.glob(os.path.join(data_train_path, "imgs") + "/*.*")
    pbar = tqdm.tqdm(files)
    for i in pbar:
        conut.append(np.min(np.array(Image.open(i))))
    plt.hist(conut)
    plt.show()
