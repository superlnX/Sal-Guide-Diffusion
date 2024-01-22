import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
from basicsr.utils import tensor2img
from omegaconf import OmegaConf

from ldm.data.dataset_salence import dataset_fi_salence
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.adapter import Adapter
from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--bsize",
    type=int,
    default=2,
    help="the prompt to render"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="the prompt to render"
)
parser.add_argument(
    "--use_shuffle",
    type=bool,
    default=True,
    help="the prompt to render"
)
parser.add_argument(
    "--dpm_solver",
    action='store_true',
    help="use dpm_solver sampling",
)
parser.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)
parser.add_argument(
    "--auto_resume",
    action='store_true',
    default=True,
    help="use plms sampling",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="ckp/sd-v1-5.ckpt",
    help="path to checkpoint of model",
)
parser.add_argument(
    "--config",
    type=str,
    default="config/train_saliency.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--print_fq",
    type=int,
    default=100,
    help="path to config which constructs model",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=1,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    '--launcher',
    default='pytorch',
    type=str,
    help='node rank for distributed training'
)
opt = parser.parse_args()

if __name__ == '__main__':
    config = OmegaConf.load(f"{opt.config}")
    opt.name = config['name']
    torch.backends.cudnn.benchmark = True
    device = 'cuda:0'
    torch.cuda.set_device(device=device)
    data_path = 'dataset/FI'
    mood_list = os.listdir(data_path)
    test_dataset = dataset_fi_salence(root_path_im='dataset/FI',
                                      root_path_sal='dataset/FI_sal',
                                      mood_list=mood_list,
                                      image_size=512,
                                      ratio=[0, 1],
                                      )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=False)

    model = load_model_from_config(config, f"{opt.ckpt}").to(device)

    # adapter
    model_ad = Adapter(cin=int(3 * 64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True,
                       use_conv=False).to(device)

    experiments_root = osp.join('experiments', opt.name)
    model_path = r'/data/linxr/my_code/科研/T2I-lxr/experiments/train_salence/models/model_ad_500.pth'
    model_ad.load_state_dict(torch.load(model_path))
    epoch = 500  # custom

    os.mkdir(os.path.join(experiments_root, 'test'))
    os.mkdir(os.path.join(experiments_root, 'test', 'ada'))
    os.mkdir(os.path.join(experiments_root, 'test', 'org'))

    save_mood = np.array([])
    for i, data in enumerate(test_dataloader):
        with torch.no_grad():
            if opt.dpm_solver:
                sampler = DPMSolverSampler(model)
            elif opt.plms:
                sampler = PLMSSampler(model)
            else:
                sampler = DDIMSampler(model)
            c = model.get_learned_conditioning(data['sentence'])
            sal = data['sal'].to(device)
            im = data['im'].to(device)
            im_sal = tensor2img(sal)
            im = tensor2img(im)
            cv2.imwrite(os.path.join(experiments_root, 'test', 'sal', 'sal_%04d.png' % i), im_sal)
            cv2.imwrite(os.path.join(experiments_root, 'test', 'im', 'org_%04d.jpg' % i), im)
            features_adapter = model_ad(sal)
            # init x_T
            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            C, H, W = shape
            size = (1, C, H, W)
            x_T = torch.randn(size, device=device)

            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                             conditioning=c,
                                             batch_size=opt.n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=opt.scale,
                                             unconditional_conditioning=model.get_learned_conditioning(
                                                 opt.n_samples * [""]),
                                             eta=opt.ddim_eta,
                                             x_T=x_T,
                                             features_adapter=features_adapter)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            mood_str = data['sentence'][0].split(" ")[-1]
            save_mood = np.append(save_mood, mood_str)
            for id_sample, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * x_sample
                img = x_sample.astype(np.uint8)
                cv2.imwrite(os.path.join(experiments_root, 'test', 'ada',
                                         'sample_ada_' + str(i) + '_' + mood_str + '.png'), img[:, :, ::-1])

            samples_ddim_org, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=model.get_learned_conditioning(
                                                     opt.n_samples * [""]),
                                                 eta=opt.ddim_eta,
                                                 x_T=x_T)
            x_samples_ddim_org = model.decode_first_stage(samples_ddim_org)
            x_samples_ddim_org = torch.clamp((x_samples_ddim_org + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim_org = x_samples_ddim_org.cpu().permute(0, 2, 3, 1).numpy()
            for id_sample, x_sample in enumerate(x_samples_ddim_org):
                x_sample = 255. * x_sample
                img = x_sample.astype(np.uint8)
                cv2.imwrite(os.path.join(experiments_root, 'test', 'org',
                                         'sample_org_' + str(i) + '_' + mood_str + '.png'), img[:, :, ::-1])

    np.save(os.path.join(experiments_root, 'test'), 'mood.npy')
