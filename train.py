# pix2pix 🚀 by Deeper
import argparse
import time
import datetime
import sys

from models import *

from datasets import *

from tensorboardX import SummaryWriter

from jimm import tf_efficientnet_b5, tf_efficientnet_b6, vit_base_patch16_224_in21k, vit_base_patch16_384,swin_base_patch4_window7_224_in22k
import warnings

warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--input_path", type=str, default="/data/comptition/jittor/data/train")
parser.add_argument("--output_path", type=str, default="./results/")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=384, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs(f"{opt.output_path}/images/", exist_ok=True)
os.makedirs(f"{opt.output_path}/saved_models/", exist_ok=True)

writer = SummaryWriter(opt.output_path)

# Loss functions
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_pixelwise = nn.L1Loss()
criterion_loss = nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100
lambda_FM = 50

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
generator = UnetGenerator(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)

discriminator = NLayerDiscriminator()

if opt.epoch != 0:
    # Load pretrained models
    generator.load(f"{opt.output_path}/saved_models/generator_{opt.epoch}.pkl")
    discriminator.load(f"{opt.output_path}/saved_models/discriminator_{opt.epoch}.pkl")

# Optimizers
optimizer_G = jt.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = jt.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

G_scheduler = jt.lr_scheduler.MultiStepLR(optimizer=optimizer_G, milestones=[100,300], gamma=0.1)
D_scheduler = jt.lr_scheduler.MultiStepLR(optimizer=optimizer_D, milestones=[100,300], gamma=0.1)

# Configure dataloaders
transforms = [
    transform.Resize(size=(640, 640), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

dataloader = ImageDataset(opt.input_path, mode="train", transforms=transforms).set_attrs(
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

warmup_times = -1
run_times = 3000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------

prev_time = time.time()

vit = vit_base_patch16_384(pretrained=True)
resnet = tf_efficientnet_b5(pretrained=True)

for epoch in range(opt.epoch, opt.n_epochs):

    if epoch ==160:
        print('160~200')
        resnet = tf_efficientnet_b6(pretrained=True)
    elif epoch == 200:
        print('>=200')
        vit = vit_base_patch16_224_in21k(pretrained=True)
        resnet = swin_base_patch4_window7_224_in22k(pretrained=True)

    for i, (real_B, real_A, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = jt.ones([real_A.shape[0], 1]).stop_grad()
        fake = jt.zeros([real_A.shape[0], 1]).stop_grad()
        fake_B = generator(real_A)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        start_grad(discriminator)
        fake_AB = jt.contrib.concat((real_A, fake_B), 1)
        pred_fake = discriminator(fake_AB.detach())
        loss_D_fake = criterion_GAN(pred_fake, False)
        real_AB = jt.contrib.concat((real_A, real_B), 1)
        pred_real = discriminator(real_AB)
        loss_D_real = criterion_GAN(pred_real, True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        optimizer_D.step(loss_D)
        writer.add_scalar('train/loss_D', loss_D.item(), epoch * len(dataloader) + i)

        # ------------------
        #  Train Generators
        # ------------------
        stop_grad(discriminator)
        fake_AB = jt.contrib.concat((real_A, fake_B), 1)
        pred_fake = discriminator(fake_AB)
        loss_G_GAN = criterion_GAN(pred_fake, True)
        loss_G_L1 = criterion_pixelwise(fake_B, real_B)

        if epoch >=160 and epoch <200:
            fake_B_1 = jt.nn.interpolate(fake_B, size=(528, 528), mode='bilinear', align_corners=False)
            real_B_1 = jt.nn.interpolate(real_B, size=(528, 528), mode='bilinear', align_corners=False)

            fake_B_2 = jt.nn.interpolate(fake_B, size=(384, 384), mode='bilinear', align_corners=False)
            real_B_2 = jt.nn.interpolate(real_B, size=(384, 384), mode='bilinear', align_corners=False)
        elif epoch >=200:
            fake_B_1 = jt.nn.interpolate(fake_B, size=(224, 224), mode='bilinear', align_corners=False)
            real_B_1 = jt.nn.interpolate(real_B, size=(224, 224), mode='bilinear', align_corners=False)

            fake_B_2 = jt.nn.interpolate(fake_B, size=(224, 224), mode='bilinear', align_corners=False)
            real_B_2 = jt.nn.interpolate(real_B, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            fake_B_1 = jt.nn.interpolate(fake_B, size=(456, 456), mode='bilinear', align_corners=False)
            real_B_1 = jt.nn.interpolate(real_B, size=(456, 456), mode='bilinear', align_corners=False)

            fake_B_2 = jt.nn.interpolate(fake_B, size=(384, 384), mode='bilinear', align_corners=False)
            real_B_2 = jt.nn.interpolate(real_B, size=(384, 384), mode='bilinear', align_corners=False)

        real_features_1, fake_features_1 = resnet(real_B_1), resnet(fake_B_1)
        real_features_2, fake_features_2 = vit(real_B_2), vit(fake_B_2)

        loss_FM = criterion_loss(real_features_1, fake_features_1) + criterion_loss(real_features_2, fake_features_2)

        loss_G = loss_G_GAN + lambda_pixel * loss_G_L1 + loss_FM * lambda_FM / 2.0
        optimizer_G.step(loss_G)
        writer.add_scalar('train/loss_G', loss_G.item(), epoch * len(dataloader) + i)

        jt.sync_all(True)

        if jt.rank == 0:
            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            jt.sync_all()
            if batches_done % 5 == 0:
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.numpy()[0],
                        loss_G.numpy()[0],
                        loss_G_L1.numpy()[0],
                        loss_G_GAN.numpy()[0],
                        time_left,
                    )
                )

    G_scheduler.step()
    D_scheduler.step()

    if jt.rank == 0 and opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
        generator.save(os.path.join(f"{opt.output_path}/saved_models/generator_{epoch}.pkl"))
        discriminator.save(os.path.join(f"{opt.output_path}/saved_models/discriminator_{epoch}.pkl"))
