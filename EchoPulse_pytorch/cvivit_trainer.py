import pdb
from math import sqrt
from random import choice
from pathlib import Path
from shutil import rmtree
import webdataset as wds
import torchvision.io
import io
import json
import os
import pickle
import re
import tempfile
from einops import rearrange
import time

from beartype import beartype

import torch
import glob
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from einops import rearrange

from EchoPulse_pytorch.optimizer import get_optimizer, LinearWarmup_CosineAnnealing

from ema_pytorch import EMA

from EchoPulse_pytorch.cvivit import CViViT
from EchoPulse_pytorch.data import ImageDataset, VideoDataset, video_tensor_to_gif, video_to_tensor, video_tensor_to_pil_first_image
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize, CenterCrop, RandomHorizontalFlip

from accelerate import Accelerator

import wandb

# helpers


def exists(val):
    return val is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


def noop(*args, **kwargs):
    pass


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# main trainer class


@beartype
class CViViTTrainer(nn.Module):
    def __init__(
        self,
        vae,
        *,
        num_train_steps,
        batch_size,
        folder,
        num_frames=11,
        lr=3e-4,
        grad_accum_every=1,
        wd=0.,
        max_grad_norm=0.5,
        train_on_images=False,
        force_cpu=False,
        wandb_mode="disabled",
        discr_max_grad_norm=None,
        linear_warmup_start_factor=0.1,
        linear_warmup_total_iters=100,
        cosine_annealing_T_max=1000000,
        cosine_annealing_eta_min=1e-5,
        save_results_every=1000,
        save_model_every=1000,
        results_folder='./results',
        scheduler_optim_overhead=0,
        valid_frac=0.05,
        random_split_seed=42,
        use_ema=True,
        ema_beta=0.995,
        ema_update_after_step=0,
        ema_update_every=1,
        apply_grad_penalty_every=4,
        inference=False,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        # Initialize the attribute for custom objects
        self._custom_objects = []
        image_size = vae.image_size

        # prepare dataset and valid_dataset
        # TO DO : split into a function ?

        n_tar = len(glob.glob(folder + "/*.tar")) - 1
        str_n_tar = str(n_tar).zfill(5)
        valid_n_tar = int(n_tar / 100 * 10)
        train_n_tar = n_tar - valid_n_tar - 1
        str_train_n_tar = str(train_n_tar).zfill(5)

        train_url = folder + "/{00000.." + str_train_n_tar + "}.tar"
        valid_url = folder + \
            "/{" + str_train_n_tar + ".." + str_n_tar + "}.tar"

        print("Train_url : ", train_url)
        print("Valid_url : ", valid_url)
        print("Batch size : ", batch_size)

        # Training on images
        if (train_on_images):
            transforms = Compose([Resize(image_size), CenterCrop(
                image_size), RandomHorizontalFlip(), ToTensor()])

            self.ds = wds.WebDataset(train_url)
            self.ds.decode("pil").to_tuple(
                "jpg").map_tuple(transforms).shuffle(1000)

            self.valid_ds = wds.WebDataset(valid_url)
            self.valid_ds.decode("pil").to_tuple(
                "jpg").map_tuple(transforms).shuffle(1000)

        # Training on videos
        else:

            # def decode_and_transform(key, data):

            #     extension = re.sub(r".*[.]", "", key)
            #     if extension not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
            #         return None

            #     with tempfile.TemporaryDirectory() as dirname:
            #         fname = os.path.join(dirname, f"file.{extension}")
            #         with open(fname, "wb") as stream:
            #             stream.write(data)
            #         video = torchvision.io.read_video(fname, pts_unit="sec")[0]

            #         # video = rearrange(video, 'f h w c -> f c h w')
            #         # _, _, h, w = video.shape
            #         # to_crop = min(h, w)
            #         # video = CenterCrop(to_crop)(video)
            #         # video = Resize(image_size, antialias=True)(video)
            #         # video = rearrange(video, 'f c h w -> c f h w')

            #         video = rearrange(video, 'f h w c -> c f h w')

            #         video = video / 255.0
            #         return video

            # def my_split_by_node(urls):
            #     node_id, node_count = torch.distributed.get_rank(), torch.distributed.get_world_size()
            #     urls = list(urls)
            #     return urls[node_id::node_count]

            # if (force_cpu) or torch.cuda.device_count() == 1:
            #     self.ds = wds.WebDataset(train_url)
            #     self.ds.decode(decode_and_transform).to_tuple(
            #         "mp4.mp4").shuffle(1000)
            #     # .with_length(48)

            #     self.valid_ds = wds.WebDataset(
            #         valid_url)
            #     self.valid_ds.decode(decode_and_transform).to_tuple(
            #         "mp4.mp4").shuffle(1000)
            #     # .with_length(48)
            # else:
            #     self.ds = wds.WebDataset(
            #         train_url, nodesplitter=my_split_by_node)
            #     self.ds.decode(decode_and_transform).to_tuple(
            #         "mp4.mp4").shuffle(1000)
            #     # .with_length(48)

            #     self.valid_ds = wds.WebDataset(
            #         valid_url, nodesplitter=my_split_by_node)
            #     self.valid_ds.decode(decode_and_transform).to_tuple(
            #         "mp4.mp4").shuffle(1000)
            #     # .with_length(48)
            self.ds = VideoDataset(folder, image_size,sample_texts='/raid/home/CAMCA/yl463/Video/sorted_files.txt', num_frames = num_frames)
            
        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')
       
        # wandb config
        config = {}
        arguments = locals()
        for key in arguments.keys():
            if key not in ['self', 'config', '__class__', 'vae']:
                config[key] = arguments[key]

        # 3. Log gradients and model parameters
        # if (wandb_mode != "disabled"):
        #    wandb.watch(vae, log='all', log_freq=3)

        if not inference:
            self.wandb_mode = wandb_mode
        else:
            self.wandb_mode = 'disabled'

        from accelerate.utils import DistributedDataParallelKwargs
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(cpu=force_cpu, log_with="wandb", kwargs_handlers=[kwargs])
        
        self.accelerator.init_trackers(project_name="CViViT", config=config, init_kwargs={"wandb": {"mode": self.wandb_mode, "config": config}})

        if self.accelerator.is_main_process:
            print('config\n')
            print(config)        
        self.wandb_mode = wandb_mode
        self.vae = vae
        self.vae.wandb_mode = wandb_mode
        self.use_discr = vae.use_discr

        self.use_ema = use_ema
        if self.is_main and use_ema:
            self.ema_vae = EMA(
                vae, update_after_step=ema_update_after_step, update_every=ema_update_every)

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        all_parameters = list(vae.parameters())

        non_vae_parameters = list(vae.i3d.parameters()) + \
            list(vae.loss_fn_lpips.parameters())
        if vae.use_discr:
            discr_parameters = list(vae.discr.parameters())
            non_vae_parameters = non_vae_parameters + discr_parameters

        vae_parameters = []
        for param in all_parameters:
            if param not in set(non_vae_parameters):
                vae_parameters.append(param)

        self.vae_parameters = vae_parameters

        self.optim = get_optimizer(vae_parameters, lr=lr, wd=wd)
        self.scheduler_optim = LinearWarmup_CosineAnnealing(optimizer=self.optim, linear_warmup_start_factor=linear_warmup_start_factor,
                                                            linear_warmup_total_iters=linear_warmup_total_iters, cosine_annealing_T_max=cosine_annealing_T_max, cosine_annealing_eta_min=cosine_annealing_eta_min)
        self.scheduler_optim_overhead = scheduler_optim_overhead
        if vae.use_discr:
            self.discr_optim = get_optimizer(
                discr_parameters, lr=1e-4, wd=1e-4)

        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # dataloader

        self.dl = DataLoader(
            self.ds,
            # self.ds.batched(batch_size),
            batch_size=batch_size,
            shuffle=True
        )

        self.valid_dl = DataLoader(
            self.valid_ds,  ##self.valid_ds
            # self.valid_ds.batched(batch_size),
            batch_size=batch_size,
            shuffle=True
        )
        # prepare with accelerator

        if vae.use_discr:
            (
                self.vae,
                self.optim,
                self.discr_optim,
                self.dl,
                self.valid_dl
            ) = self.accelerator.prepare(
                self.vae,
                self.optim,
                self.discr_optim,
                self.dl,
                self.valid_dl
            )

        else:

            (
                self.vae,
                self.optim,
                self.dl,
                self.valid_dl
            ) = self.accelerator.prepare(
                self.vae,
                self.optim,
                self.dl,
                self.valid_dl
            )
        
        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

    # def save(self, path):

    #     self.accelerator.save_state(path)

    #     return
    def save(self, path):
        # Ensure the save path directory exists
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save the state dict of the model
        model_file = path / 'pytorch_model.bin'
        torch.save(self.vae.state_dict(), model_file)

        # If using EMA, save its state dict as well
        if self.use_ema:
            ema_model_file = path / 'ema_pytorch_model.bin'
            torch.save(self.ema_vae.ema_model.state_dict(), ema_model_file)

        # Save optimizer and other training states if needed
        optimizer_file = path / 'optimizer.bin'
        torch.save({
            'optimizer': self.optim.state_dict(),
            'steps': self.steps.item()
        }, optimizer_file)

        return
    def load(self, path):
        model_file = os.path.join(path, 'pytorch_model.bin')  # Specify the model file
        
        # Load the state_dict from the specified file
        state_dict = torch.load(model_file, map_location=self.accelerator.device)
        
        # Remove the 'step' key if it exists
        if 'step' in state_dict:
            del state_dict['step']
        
        # Load the state_dict into the model
        self.vae.load_state_dict(state_dict)

    def print(self, msg):
        print(msg) ##To-dos
        # self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())
        apply_grad_penalty = not (steps % self.apply_grad_penalty_every)

        self.vae.train()

        # logs

        logs = {}

        # update vae (generator)
        # time_cvivit = time.time()
        for _ in range(self.grad_accum_every):
            img = next(self.dl_iter)[0]
            img = img.to(device)

            with self.accelerator.autocast():
                loss = self.vae(
                    img,
                    apply_grad_penalty=apply_grad_penalty,
                    accelerator_tracker=self.accelerator
                )

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

            self.accelerator.log({"vae_loss": loss.item()})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(
                self.vae.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        self.scheduler_optim.step(self.steps + self.scheduler_optim_overhead)
        self.accelerator.log({"lr": self.optim.param_groups[0]["lr"]})

        # update discriminator
        # DISCRIMINATOR IS NOT TRAINED ON THE SAME DATA AS THE VAE

        if self.use_discr:
            self.discr_optim.zero_grad()

            for _ in range(self.grad_accum_every):
                img = next(self.dl_iter)[0]
                img = img.to(device)

                loss = self.vae(img, return_discr_loss=True)

                self.accelerator.backward(loss / self.grad_accum_every)

                accum_log(
                    logs, {'discr_loss': loss.item() / self.grad_accum_every})

                self.accelerator.log({"discr_loss": loss.item()})

            if exists(self.discr_max_grad_norm):
                self.accelerator.clip_grad_norm_(
                    self.vae.discr.parameters(), self.discr_max_grad_norm)

            self.discr_optim.step()

            # log

            self.print(
                f"{steps}: vae loss: {logs['loss']} - discr loss: {logs['discr_loss']}")

        self.print(f"{steps}: vae loss: {logs['loss']}")

        # update exponential moving averaged generator

        if self.is_main and self.use_ema:
            self.ema_vae.update()

        # sample results every so often

        if (self.steps == 0):
            self.valid_data_to_log = next(self.valid_dl_iter)[0]

        if not (steps % self.save_results_every):
            vaes_to_evaluate = ((self.vae, str(steps)),)

            if self.use_ema:
                vaes_to_evaluate = (
                    (self.ema_vae.ema_model, f'{steps}.ema'),) + vaes_to_evaluate

            for model, filename in vaes_to_evaluate:
                model.eval()

                valid_data = next(self.valid_dl_iter)[0]

                is_video = valid_data.ndim == 5

                valid_data = valid_data.to(device)

                recons = model(self.valid_data_to_log, return_recons_only=True)

                # if is video, save gifs to folder
                # else save a grid of images

                if is_video:
                    sampled_videos_path = self.results_folder / \
                        f'samples.{filename}'
                    (sampled_videos_path).mkdir(parents=True, exist_ok=True)

                    for i, tensor in enumerate(recons.unbind(dim=0)):
                        video_tensor_to_gif(tensor.cpu(), str(
                            sampled_videos_path / f'{filename}-{i}.gif'))

                        if (i < 4):
                            self.accelerator.log({
                                f"image{i}": [wandb.Image(video_tensor_to_pil_first_image(tensor.cpu())), wandb.Image(video_tensor_to_pil_first_image(self.valid_data_to_log[i].cpu()))],
                            })


                else:
                    imgs_and_recons = torch.stack((valid_data, recons), dim=0)
                    imgs_and_recons = rearrange(
                        imgs_and_recons, 'r b ... -> (b r) ...')

                    imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                    grid = make_grid(imgs_and_recons, nrow=2,
                                     normalize=True, range=(0, 1))

                    logs['reconstructions'] = grid

                    save_image(
                        grid, str(self.results_folder / f'{filename}.png'))

            self.print(f'{steps}: saving to {str(self.results_folder)}')

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):

            save_path = Path(
                str(self.results_folder / f'ckpt_accelerate_{steps}/'))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save(save_path)

            self.print(f'{steps}: saving model to {str(save_path)}')

        self.steps += 1
        return logs

    def train(self, log_fn=noop):
        # device = next(self.vae.parameters())[0].device

        while self.steps < self.num_train_steps:
            timestep_wallclock_time = time.time()
            logs = self.train_step()
            log_fn(logs)
            print(time.time()-timestep_wallclock_time,
                  'time taken to perform one timestep')

        self.print('training complete')
