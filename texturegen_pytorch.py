# Std. lib imports
import os
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
from argparse import ArgumentParser

# 3rd-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.functional import F
import torchvision.io as io
from torch.nn.modules.pooling import FractionalMaxPool2d, LPPool2d
from PIL import Image
import matplotlib.pyplot as plt



def parse_res_str(res: str) -> List[int]:
    return list(map(int, res.replace(' ', "").lower().split('x')))


def res_str(res: List[int]) -> str:
    return f"{res[0]}x{res[1]}"


argparser = ArgumentParser()
argparser.description = "Something something experimental texture image generator ... ."
argparser.usage = f"'python3 {Path(__file__).name} -i {os.sep}path{os.sep}to{os.sep}image.png' [--help]"


DEFAULT_PARAMS: Dict[str, Any] = {"out_shape": [512, 512],
                                  "patch_shape": [32, 32],
                                  "batchsize": 64,
                                  "stacking_size": 3,
                                  "lr_init_d": 0.004,
                                  "lr_init_g": 0.0015,
                                  "seed": None,
                                  "n_checkpoint": 256,
                                  "torch_compile": False,
                                  "rot_invariant": False,
                                  "aug_flip_xy": False,
                                  }


argparser.add_argument("-i", "--input", type=str, required=True, help="Image file to generate textures from")
argparser.add_argument("-r", "--res", type=str, default=res_str(DEFAULT_PARAMS["out_shape"]), help="Output image resolution, e.g. '512x512' (width x height)")
argparser.add_argument("-ps", "--patch-size", type=str, default=res_str(DEFAULT_PARAMS["patch_shape"]), help="The sampling patch size, e.g. '32x32' (width x height)")
argparser.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_PARAMS["batchsize"], help="The per-GPU batch size to use")
argparser.add_argument("-ss", "--stack-size", type=int, default=DEFAULT_PARAMS["stacking_size"], help="Number of samples to compare each iteration. 2-3 recommended.")
argparser.add_argument("-s", "--seed", type=int, default=DEFAULT_PARAMS["seed"], help="Make the texture generator deterministic with a seed number of choice")
argparser.add_argument("-lr-d", "--learning-rate-discriminator", type=float, default=DEFAULT_PARAMS["lr_init_d"], help="Initial learning-rate for discriminator network")
argparser.add_argument("-lr-g", "--learning-rate-generator", type=float, default=DEFAULT_PARAMS["lr_init_g"], help="Initial learning-rate for generator network")
#argparser.add_argument("-l", "--log", action="store_true", help="Log results to disk")
#argparser.add_argument("-l", "--log", type=str, default=None, help="Name of file to log results to (filename will be extended with a UTC datetime prefix)")
argparser.add_argument("--save-interval", type=int, default=DEFAULT_PARAMS["n_checkpoint"], help="Save training progress every ith step")
argparser.add_argument("-c", "--compile", action="store_true", help="Use Torch Compile for faster learning")
argparser.add_argument("--device", type=int, default=0, help="Index of GPU to use. For multi-GPU machines (0, 1, ..., k)")
argparser.add_argument("--a", "--activation", type=str, default="relu", help="Name of activation function between the convolution layers to try")

# ------ A few untested ideas yet to be discussed ------
#
argparser.add_argument("-ri", "--rotation-invariant", action="store_true", help="Features in the input image can be generated at any angle (0-360 degrees)")  # TODO: Try randomly rotated patches?
argparser.add_argument("-f", "--flip", type=str, default=DEFAULT_PARAMS["n_checkpoint"], help="Add horizontally and/or vertically flipped image features for extra data, e.g 'x' for horizontal, 'xy' for both axes")


args = argparser.parse_args()

OUTPUT_SHAPE = parse_res_str(args.res)[::-1]  # WxH -> HxW
PATCH_SHAPE = parse_res_str(args.patch_size)[::-1]  # WxH -> HxW
BATCH_SIZE = args.batch_size
STACKING_SIZE = args.stack_size
LEARNING_RATE_D = args.learning_rate_discriminator
LEARNING_RATE_G = args.learning_rate_generator
SAVE_INTERVAL = args.save_interval
SRC_IMAGE = Path(args.input)
PRINT_TIME = 5000
TORCH_COMPILE = args.compile
SEED: int = args.seed
device_i: int = args.device


if not SRC_IMAGE.exists():
    print(f"Input image file '{args.input}' not found.", file=sys.stderr)
    exit(1)


if SEED is not None:
    # import random
    # random.seed(SEED)
    # Ref. https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"RNG seed is {SEED} (numpy, torch)")


IS_COLAB = 'google.colab' in sys.modules
print(f"IS_COLAB: {IS_COLAB}")

IMGNAME: str = f"{SRC_IMAGE.name.split('.')[0]}_res{args.res}_patch{args.patch_size}_bs{BATCH_SIZE}_seed{SEED}_stacking{STACKING_SIZE}_gpu{device_i}_relu"
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/gdrive')
    imgfilename = f"/content/gdrive/My Drive/texgen/input/{SRC_IMAGE}"
else:
    imgfilename = SRC_IMAGE

real_img = Image.open(imgfilename)
real_img = transforms.ToTensor()(real_img)[:3, :, :]  # Strip alpha-channel if present
real_img = real_img*2.0-1.0
real_img = real_img.unsqueeze(0).to(device)

patch_unfold = nn.Unfold(kernel_size=(PATCH_SHAPE[0], PATCH_SHAPE[1]))
real_patch_unfold = nn.Unfold(kernel_size=(PATCH_SHAPE[0], PATCH_SHAPE[1]))
real_img = real_patch_unfold(real_img)
print(real_img.shape, real_img.dtype)

def realimg():
    output_indices = torch.randint(0, real_img.shape[2], (BATCH_SIZE*STACKING_SIZE,))
    output = real_img[:, :, output_indices]
    output = output.transpose(1,2).contiguous()
    output = output.view(BATCH_SIZE,3*output.shape[1]//BATCH_SIZE,PATCH_SHAPE[0], PATCH_SHAPE[1])
    return output

class FakeImg(nn.Module):
    def __init__(self):
        super(FakeImg, self).__init__()
        self.img = nn.Parameter(torch.zeros(1, 3, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1]).to(device))

    def forward(self):
        processed_img = self.img
        processed_img = torch.cat([processed_img, processed_img[:, :, :PATCH_SHAPE[0] - 1, :]], dim=2)
        processed_img = torch.cat([processed_img, processed_img[:, :, :, :PATCH_SHAPE[1] - 1]], dim=3)
        
        #img_crop_x = torch.randint(0, PATCH_SHAPE[0], ())
        #img_crop_y = torch.randint(0, PATCH_SHAPE[1], ())
        #processed_img = processed_img[:, :, img_crop_y:, img_crop_x:]
        
        output = patch_unfold(processed_img)
        output_indices = torch.randint(0, output.shape[2], (BATCH_SIZE*STACKING_SIZE,))
        output = output[:, :, output_indices]
        output = output.transpose(1,2).contiguous()
        output = output.view(BATCH_SIZE,3*output.shape[1]//BATCH_SIZE,PATCH_SHAPE[0], PATCH_SHAPE[1]).contiguous()
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=3 * STACKING_SIZE, out_channels=24 * 4, kernel_size=3, padding='same'),
            nn.Conv2d(in_channels=24 * 4, out_channels=32 * 4, kernel_size=3, padding='same'),
            nn.Conv2d(in_channels=32 * 4, out_channels=64 * 4, kernel_size=3, padding='same')
        ])
        
        self.convs2 = nn.ModuleList([
            nn.Conv2d(in_channels=24 * 4, out_channels=24 * 4, kernel_size=3, padding='same'),
            nn.Conv2d(in_channels=32 * 4, out_channels=32 * 4, kernel_size=3, padding='same'),
            nn.Conv2d(in_channels=64 * 4, out_channels=64 * 4, kernel_size=3, padding='same')
        ])
        
        self.lns = nn.ModuleList([nn.LayerNorm(24 * 4),
                                  nn.LayerNorm(32 * 4),
                                  nn.LayerNorm(64 * 4)])
        
        self.lns2 = nn.ModuleList([nn.LayerNorm(24 * 4),
                                   nn.LayerNorm(32 * 4),
                                   nn.LayerNorm(64 * 4)])
        
        self.pools = nn.ModuleList([nn.AvgPool2d(kernel_size=2),
                                    nn.AvgPool2d(kernel_size=2),
                                    None])
        
        self.lastdense = nn.Linear(64 * 4 * (PATCH_SHAPE[0] // 4) * (PATCH_SHAPE[1] // 4), 1, bias=False)
                
    def do_layernorm(self, tensor, lnname):
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
        tensor = lnname(tensor)
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
        return tensor
                
                
    def forward(self, inputdata):
        for n in range(3):
            inputdata = F.relu(self.convs[n](inputdata))
            inputdata = self.do_layernorm(inputdata, self.lns2[n])#self.lns2[n](inputdata)
            inputdata = inputdata + F.relu(self.convs2[n](inputdata))
            inputdata = self.do_layernorm(inputdata, self.lns[n])#self.lns[n](inputdata)
            if self.pools[n] is not None:
                inputdata = self.pools[n](inputdata)
                #print(inputdata.shape)
            
        inputdata = inputdata.view(inputdata.size(0), -1)
        inputdata = self.lastdense(inputdata)
        inputdata = inputdata.squeeze(1)
        return inputdata

fakeimg = FakeImg().to(device)
d = Discriminator().to(device)

optimizer_d = optim.Adam(d.parameters(), lr=LEARNING_RATE_D, amsgrad=True)
optimizer_g = optim.Adam(fakeimg.parameters(), lr=LEARNING_RATE_G, amsgrad=True)

iters = 0

def do_thing_D():
    with torch.no_grad():
        fi = fakeimg()
        ri = realimg()

    fakes = d(fi)
    reals = d(ri)
    
    reals = reals.unsqueeze(0)
    fakes = fakes.unsqueeze(1)
    return fakes - reals

def train_D():
    # train discriminator
    optimizer_d.zero_grad()
    loss = F.softplus(do_thing_D())
    loss.mean().backward()
    optimizer_d.step()

def do_thing_G():
    with torch.no_grad():
        ri = realimg()

    fakes = d(fakeimg())
    reals = d(ri)
    
    reals = reals.unsqueeze(0)
    fakes = fakes.unsqueeze(1)
    return reals - fakes

def train_G():
    # train generator
    optimizer_g.zero_grad()
    loss = F.relu(do_thing_G())
    loss.mean().backward()
    optimizer_g.step()


if TORCH_COMPILE:
    train_D_opt = torch.compile(train_D)
    train_G_opt = torch.compile(train_G)
else:
    train_D_opt = train_D
    train_G_opt = train_G

try:
    currtime = time.time()
    curriters = 0
    while 1:
        iters += 1
        curriters += 1
        train_D_opt()
        if iters >= 64:
            train_G_opt()

        if (time.time() - currtime) * 1000.0 > PRINT_TIME:
            delta = time.time() - currtime
            
            print(f"#{iters}, {delta * 1000.0 / curriters} ms/iter")

            currtime = time.time()
            curriters = 0
        
        if iters % SAVE_INTERVAL == 0:
            img = (fakeimg.img.squeeze(0) + 1.0) * 127.5
            img = torch.clamp(img, 0.0, 255.0).to("cpu")
            img = img.byte()
            
            if IS_COLAB:
                io.write_png(img, f"/content/gdrive/My Drive/texgen/{IMGNAME}_gen{iters}.png")
            else:
                outpath: Path = Path(f"outputs/{IMGNAME}_gen{iters}.png")
                os.makedirs(outpath.parent, exist_ok=True)
                io.write_png(img, str(outpath))
except KeyboardInterrupt:
    print("Run aborted by user")
