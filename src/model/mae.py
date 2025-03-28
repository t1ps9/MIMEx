import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm.optim.optim_factory as optim_factory
import wandb

from ..utils.logger import WandbLogger
from .mae_models import mae_vit_mini_patch8
from .mae_transforms import random_resized_crop, horizontal_flip


class DataAug(nn.Module):
    def __init__(
        self, out_size, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation='bicubic', flip_prob=0.5):
        super().__init__()
        self.out_size = out_size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.flip_prob = flip_prob

    def forward(self, x):
        b, t, c, h, w = x.shape
        assert h == w

        x_out = torch.zeros(
            b, t, c, self.out_size, self.out_size, device=x.device)
        for bid in range(b):
            a = random_resized_crop(
                x[bid], self.out_size, self.out_size, self.scale, self.ratio,
                interpolation=self.interpolation)
            a = horizontal_flip(a, self.flip_prob)
            x_out[bid] = a

        return x_out


def get_data_aug(out_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            out_size, scale=(0.8, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
    ])


def save_image(image, save_dir='/home/', fn='test'):
    assert image.shape[2] == 3
    im = torch.clip(image.detach().cpu() * 255, 0, 255).int().numpy().astype(
        np.uint8)
    plt.imshow(im)
    plt.title('', fontsize=16)
    plt.axis('off')
    plt.imsave(os.path.join(save_dir, f'{fn}.png'), im)


class MAE:
    def __init__(
        self, model_cfg, input_size=96, mask_ratio=0.9, num_mask_samples=10,
        weight_decay=0.05, base_lr=1e-3, save_vis=False, save_dir='',
        num_vis_samples=10, use_single_frame=False, no_frame_stack=False,
        wandb_project="mimex", wandb_name=None):
        self.mask_ratio = mask_ratio
        self.num_mask_samples = num_mask_samples
        self.num_vis_samples = num_vis_samples
        self.use_single_frame = use_single_frame
        self.no_frame_stack = no_frame_stack

        self.model = mae_vit_mini_patch8(
            embed_dim=model_cfg.embed_dim,
            depth=model_cfg.depth,
            num_heads=model_cfg.num_heads,
            decoder_embed_dim=model_cfg.decoder_embed_dim,
            decoder_depth=model_cfg.decoder_depth,
            decoder_num_heads=model_cfg.decoder_num_heads).cuda()
        self.data_aug = get_data_aug(input_size)

        param_groups = optim_factory.add_weight_decay(self.model, weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups, lr=base_lr, betas=(0.9, 0.95))

        self.save_vis = save_vis
        self.save_dir = save_dir

        self.logger = WandbLogger(
            project=wandb_project,
            name=wandb_name,
            config={
                "model": {
                    "embed_dim": model_cfg.embed_dim,
                    "depth": model_cfg.depth,
                    "num_heads": model_cfg.num_heads,
                    "decoder_embed_dim": model_cfg.decoder_embed_dim,
                    "decoder_depth": model_cfg.decoder_depth,
                    "decoder_num_heads": model_cfg.decoder_num_heads,
                },
                "training": {
                    "input_size": input_size,
                    "mask_ratio": mask_ratio,
                    "num_mask_samples": num_mask_samples,
                    "weight_decay": weight_decay,
                    "base_lr": base_lr,
                    "use_single_frame": use_single_frame,
                    "no_frame_stack": no_frame_stack,
                }
            }
        )
        self.logger.log_model(self.model)

    def visualize(self, x, y, mask):
        x = x.detach().cpu()
        y = self.model.unpatchify(y)
        c = y.shape[1]

        if self.use_single_frame:
            y = torch.einsum('nchw->nhwc', y).detach().cpu()
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(
                1, 1, self.model.patch_embed.patch_size[0]**2 *c)
            mask = self.model.unpatchify(mask)
            mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
            x = torch.einsum('nchw->nhwc', x)
        else:
            t = y.shape[2]
            y = torch.einsum('ncthw->nchtw', y).detach().cpu()

            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(
                1, 1, self.model.patch_embed.patch_size[0]**2 * c * t)
            mask = self.model.unpatchify(mask)
            mask = torch.einsum('ncthw->nchtw', mask).detach().cpu()
            x = torch.einsum('ncthw->nchtw', x)

            y = y.flatten(start_dim=3, end_dim=4)
            mask = mask.flatten(start_dim=3, end_dim=4)
            x = x.flatten(start_dim=3, end_dim=4)

            y = torch.einsum('nchw->nhwc', y)
            mask = torch.einsum('nchw->nhwc', mask)
            x = torch.einsum('nchw->nhwc', x)

        im_masked = x * (1 - mask)
        im_paste = x * (1 - mask) + y * mask

        for i in range(min(y.shape[0], self.num_vis_samples)):
            im_masked = x[i] * (1 - mask[i])
            im_paste = x[i] * (1 - mask[i]) + y[i] * mask[i]
            img = torch.cat((x[i], im_masked, y[i], im_paste))
            save_image(img, save_dir=self.save_dir, fn=f'{i}')
            self.logger.log_media({
                f"reconstruction_{i}": wandb.Image(img.numpy())
            })

    def forward_model(self, x, vis=False, step=None):
        C, H, W = x.shape[-3:]
        if not self.use_single_frame:
            x = x.permute(0, 2, 1, 3, 4)

        total_loss = torch.zeros(x.shape[0], device=x.device)
        mae_loss = 0.
        self.optimizer.zero_grad()

        for _ in range(self.num_mask_samples):
            loss, y, mask = self.model.update(
                x, self.mask_ratio, keep_batch_loss=True)
            mae_loss += loss.sum()
            total_loss += loss.detach()

        mae_loss.backward()
        self.optimizer.step()

        if vis:
            self.visualize(x, y, mask)

        total_loss = total_loss / self.num_mask_samples
        self.logger.log_metrics({
            "loss/mae": mae_loss.item() / self.num_mask_samples,
            "loss/total": total_loss.mean().item(),
        }, step=step)

        return total_loss

    def update(self, obs, next_obs, vis=False, step=None):
        assert obs.shape == next_obs.shape
        B, N, H, W = obs.shape

        if self.no_frame_stack:
            obs = obs.view(B, 1, N, H, W)
            next_obs = next_obs.view(B, 1, N, H, W)
            x = torch.stack([obs, next_obs], dim=2)
        else:
            assert obs.shape[1] % 3 == 0
            T = obs.shape[1] // 3
            obs = obs.view(B, T, 3, H, W)
            next_obs = next_obs.view(B, T, 3, H, W)
            x = torch.stack([obs, next_obs], dim=2)

        x = x.float() / 255

        if self.use_single_frame:
            x = x.view(-1, 3, H, W)
        else:
            if self.no_frame_stack:
                x = x.view(B, 2, N, H, W)
            else:
                x = x.view(B * T, 2, 3, H, W)

        x = self.data_aug(x)
        loss = self.forward_model(x, vis, step).view(B, -1).mean(dim=-1, keepdim=True)

        return loss

    def finish(self):
        self.logger.finish()