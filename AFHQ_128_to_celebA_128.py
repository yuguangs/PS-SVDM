#from comet_ml import Experiment
from demixing_diffusion_pytorch import Unet, GaussianDiffusion, Trainer,DiT_models
#from demixing_diffusion_pytorch import dlanet
import torchvision
import os
import errno
import shutil
import argparse
from VQGAN.vqgan import VQModel
def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass


parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=200, type=int)
parser.add_argument('--train_steps', default=200000, type=int)
parser.add_argument('--save_folder', default='./results7', type=str)
parser.add_argument('--data_path_start', default='/seu_share2/datasets/kitti/avg-kitti/', type=str)# L
parser.add_argument('--data_path_end', default='/seu_share2/datasets/kitti/avg-kitti/', type=str)# R/seu_share2/datasets/kitti/training/image_3/ 
#parser.add_argument('--load_path', default='./results2/model.pt', type=str)
parser.add_argument('--load_path', default=None, type=str)#
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--loss_type', default='l1', type=str)


args = parser.parse_args()
print(args)
'''
VQModel = VQModel(
    in_channels=3,
    ch=128,
    out_ch=3,
    ch_mult=(1, 2, 4),
    num_res_blocks=2,
    attn_resolutions=[],
    resolution=256,
    z_channels=3,
    double_z=False,
    n_embed = 8192,
    embed_dim = 3,
    ckpt_path='/seu_share/home/luxiaobo/228465/BBDM/results/VQGAN/modelf4.ckpt', # '/home/member/syg/BBDM/results/VQGAN/modelf4.ckpt'     None
    ignore_keys=[],
    image_key="image",
    colorize_nlabels=None,
    monitor=None,
    remap=None,
    sane_index_shape=False,  # tell vector quantizer to return indices as bhw
).cuda()

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    with_time_emb=not(args.remove_time_embed),
    residual=args.residual
).cuda()

image_size =(256, 768)
modelDiT = DiT_models['DiT-B/8'](input_size=image_size)
modelDiT=modelDiT.cuda()
'''
image_size =(256, 768)
model = Unet(
    dim = 128,
    dim_mults = (1, 2, 4, 8),        #(1, 2, 4, 8, 16, 32),
    channels=3,
    with_time_emb=not(args.remove_time_embed),
    residual=args.residual
).cuda()

diffusion = GaussianDiffusion(
    model,
    #VQModel,
    image_size =image_size,#(128, 384) (256, 768)  (384, 1280)
    channels = 3,
    timesteps = args.time_steps,   # number of steps
    loss_type = args.loss_type,    # L1 or L2
    train_routine = args.train_routine,
    sampling_routine = args.sampling_routine
).cuda()

import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))
#diffusion = torch.nn.DataParallel(diffusion, device_ids=1)
#torch.cuda.set_device(1)
#diffusion = diffusion.to('cuda')


trainer = Trainer(
    diffusion,
    args.data_path_end,
    args.data_path_start,
    image_size = image_size,
    train_batch_size =2,
    train_lr = 2e-4, #3.5e-4 2e-5  2e-4
    train_num_steps = args.train_steps,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    dataset = 'train'
)

trainer.train()



























