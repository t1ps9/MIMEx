import os
import argparse
from pathlib import Path
import yaml
import torch
from tqdm import tqdm

from src.datasets.replay_buffer import ReplayBuffer
from src.model.mae import MAE
from src.utils.video import save_frames_to_video


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(config):
    # Create output directories
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / 'videos'
    video_dir.mkdir(exist_ok=True)

    # Initialize dataset
    dataset = ReplayBuffer(
        config['dataset']['config_path'],
        **config['dataset']
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        shuffle=True
    )

    # Initialize model
    model = MAE(
        config['model'],
        **config['model']['wandb']
    )

    # Training loop
    step = 0
    for epoch in range(config['trainer']['epochs']):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch in pbar:
            obs, next_obs = batch
            obs = obs.cuda()
            next_obs = next_obs.cuda()

            loss = model.update(obs, next_obs, step=step)
            epoch_loss += loss.mean().item()

            pbar.set_postfix({'loss': loss.mean().item()})
            step += 1

            # Generate visualization video periodically
            if step % config['trainer']['vis_interval'] == 0:
                with torch.no_grad():
                    obs_vis = obs[:config['trainer']['vis_batch_size']]
                    next_obs_vis = next_obs[:config['trainer']['vis_batch_size']]
                    loss, pred, mask = model.model.update(
                        obs_vis, model.mask_ratio, keep_batch_loss=True)
                    pred = model.model.unpatchify(pred)

                    # Save original and reconstructed frames
                    for i in range(min(obs_vis.shape[0], config['trainer']['vis_batch_size'])):
                        frames = []
                        for t in range(obs_vis.shape[1]):
                            frame = obs_vis[i, t].cpu()
                            frames.append(frame)
                            frame = pred[i, t].cpu()
                            frames.append(frame)

                        video_path = video_dir / f'reconstruction_step{step}_sample{i}.mp4'
                        save_frames_to_video(frames, str(video_path), fps=10)

        epoch_loss /= len(dataloader)
        print(f'Epoch {epoch} average loss: {epoch_loss:.4f}')

    model.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == '__main__':
    main()
