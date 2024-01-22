import argparse
import logging
import os
import os.path as osp

import cv2
import numpy as np
import torch
from basicsr.utils import tensor2img, scandir, get_time_str, get_root_logger, get_env_info
from basicsr.utils.options import copy_opt_file, dict2str
from omegaconf import OmegaConf

from dist_util import master_only, get_bare_model
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


@master_only
def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(experiments_root, 'models'))
    os.makedirs(osp.join(experiments_root, 'training_states'))
    os.makedirs(osp.join(experiments_root, 'visualization'))


def load_resume_state(opt):
    resume_state_path = None
    if opt.auto_resume:
        state_path = osp.join('experiments', opt.name, 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt.resume_state_path = resume_state_path

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
    return resume_state


parser = argparse.ArgumentParser()
parser.add_argument(
    "--bsize",
    type=int,
    default=1,
    help="the prompt to render"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=500,
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
    train_dataset = dataset_fi_salence(root_path_im='dataset/FI',
                                       root_path_sal='dataset/FI_sal',
                                       mood_list=mood_list,
                                       image_size=512,
                                       ratio=[0, 0.9],
                                       )
    val_dataset = dataset_fi_salence(root_path_im='dataset/FI',
                                     root_path_sal='dataset/FI_sal',
                                     mood_list=mood_list,
                                     image_size=512,
                                     ratio=[0.9, 1],
                                     )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.bsize,
        shuffle=True,
        num_workers=2,
    )
    # next(iter(train_dataloader))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=False)

    model = load_model_from_config(config, f"{opt.ckpt}").to(device)
    # sketch encoder
    model_ad = Adapter(cin=int(3 * 64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True,
                       use_conv=False).to(device)

    params = list(model_ad.parameters())
    optimizer = torch.optim.AdamW(params, lr=config['training']['lr'])

    experiments_root = osp.join('experiments', opt.name)

    # resume state
    resume_state = load_resume_state(opt)
    if resume_state is None:
        mkdir_and_rename(experiments_root)
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
    else:
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
        resume_optimizers = resume_state['optimizers']
        optimizer.load_state_dict(resume_optimizers)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')

    for epoch in range(start_epoch, opt.epochs):
        for _, data in enumerate(train_dataloader):
            current_iter += 1
            with torch.no_grad():
                c = model.get_learned_conditioning(data['sentence'])
                z = model.encode_first_stage((data['im'] * 2 - 1.).cuda(non_blocking=True))
                z = model.get_first_stage_encoding(z)

            sal = data['sal'].to(device)
            optimizer.zero_grad()
            model.zero_grad()
            features_adapter = model_ad(sal)
            l_pixel, loss_dict = model(z, c=c, features_adapter=features_adapter)
            l_pixel.backward()
            optimizer.step()

            if (current_iter + 1) % opt.print_fq == 0:
                logger.info(loss_dict)

        # save checkpoint
        save_filename = f'model_ad_{epoch + 1}.pth'
        save_path = os.path.join(experiments_root, 'models', save_filename)
        save_dict = {}
        model_ad_bare = get_bare_model(model_ad)
        state_dict = model_ad_bare.state_dict()
        for key, param in state_dict.items():
            if key.startswith('module.'):  # remove unnecessary 'module.'
                key = key[7:]
            save_dict[key] = param.cpu()
        torch.save(save_dict, save_path)
        # save state
        state = {'epoch': epoch, 'iter': current_iter + 1, 'optimizers': optimizer.state_dict()}
        save_filename = f'{epoch + 1}.state'
        save_path = os.path.join(experiments_root, 'training_states', save_filename)
        torch.save(state, save_path)

        # val
        if epoch % 10 == 0:
            tar_path = os.path.join(experiments_root, 'visualization', 'epoch' + str(epoch))
            os.mkdir(tar_path)
            # os.mkdir(os.path.join(tar_path, 'img'))
            os.mkdir(os.path.join(tar_path, 'sal'))
            os.mkdir(os.path.join(tar_path, 'ada'))
            os.mkdir(os.path.join(tar_path, 'org'))
            for i, data in enumerate(val_dataloader):
                with torch.no_grad():
                    if opt.dpm_solver:
                        sampler = DPMSolverSampler(model)
                    elif opt.plms:
                        sampler = PLMSSampler(model)
                    else:
                        sampler = DDIMSampler(model)
                    sentence = data['sentence'][-1]
                    c = model.get_learned_conditioning(sentence)
                    sal = data['sal'].to(device)
                    im = data['im'].to(device)
                    im_sal = tensor2img(sal)
                    im = tensor2img(im)
                    cv2.imwrite(
                        os.path.join(tar_path, 'sal', 'sal_%04d_%02d.png' % (epoch, i + 1)),
                        im_sal)
                    # cv2.imwrite(
                    # os.path.join(tar_path, 'img', 'im_%04d_%02d.png' % (epoch, i + 1)), im)
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
                    for id_sample, x_sample in enumerate(x_samples_ddim):
                        x_sample = 255. * x_sample
                        img = x_sample.astype(np.uint8)
                        img = cv2.putText(img.copy(), sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                          (0, 255, 0), 2)

                        cv2.imwrite(os.path.join(tar_path, 'ada',
                                                 'sample_e%04d_s%04d.png' % (epoch, i + 1)), img[:, :, ::-1])

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
                        img = cv2.putText(img.copy(), sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                          (0, 255, 0), 2)
                        cv2.imwrite(os.path.join(tar_path, 'org',
                                                 'sample_org_e%04d_s%04d.png' % (epoch, i + 1)), img[:, :, ::-1])
