import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import argparse
import os
import numpy as np
import itertools
import sys
import cv2
from models import *
from datasets import *

import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="/data/comptition/jittor/data/val_B-labels-clean")
parser.add_argument("--model_path", type=str, default="./results/71_baseline/saved_models/")
parser.add_argument("--output_path", type=str, default="./results/")
parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
opt = parser.parse_args()

generator = UnetGenerator(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)
generator.load(opt.model_path+"/generator_299.pkl")

generator_1 = UnetGenerator(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)
generator_1.load(opt.model_path+"/generator_289.pkl")

generator_2 = UnetGenerator(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)
generator_2.load(opt.model_path+"/generator_279.pkl")

transforms = [
    transform.Resize(size=(640, 640), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

test_dataloader = ImageDataset(opt.input_path, mode="val", transforms=transforms).set_attrs(
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=1,
)

@jt.single_process_scope()
def eval():
    cnt = 1
    os.makedirs(f"{opt.output_path}/", exist_ok=True)
    for i, (_, real_A, photo_id) in enumerate(test_dataloader):
        fake_B = (generator(real_A) + generator_1(real_A) + generator_2(real_A) ) / 3
        fake_B = jt.nn.interpolate(fake_B, size=(384,512), mode='bicubic', align_corners=False)
        fake_B = np.clip((fake_B + 1) / 2, 0, 1) * 255
        fake_B = fake_B.astype('uint8')
        for idx in range(fake_B.shape[0]):
            cv2.imwrite(f"{opt.output_path}/{photo_id[idx]}.jpg", fake_B[idx].transpose(1,2,0)[:,:,::-1])
            cnt += 1

eval()



