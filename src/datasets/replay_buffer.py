import datetime
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.datasets.dmc_env import DMCEnvironment
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class ReplayBufferDataset(BaseDataset):
    def __init__(
        self,
        config_path: str,
        *args,
        **kwargs
    ):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config

        self.env = DMCEnvironment(
            name=config['env']['name'],
            frame_stack=config['env']['frame_stack'],
            action_repeat=config['env']['action_repeat'],
            seed=config['env']['seed'],
            pixels_only=config['env']['pixels_only']
        )

        self.buffer = deque(maxlen=config['buffer']['max_size'])
        self.current_episode = []

        self.nstep = config['buffer']['nstep']
        self.discount = config['buffer']['discount']

        index_path = ROOT_PATH / "data" / "replay_buffer" / config['dataset']['name'] / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(config['dataset']['name'])

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name: str) -> List[Dict]:
        index = []
        data_path = ROOT_PATH / "data" / "replay_buffer" / name
        data_path.mkdir(exist_ok=True, parents=True)

        write_json(index, str(data_path / "index.json"))
        return index

    def collect_episode(self, policy_fn) -> None:
        time_step = self.env.reset()
        episode = []

        while not time_step.last():
            action = policy_fn(time_step.observation)
            time_step = self.env.step(action)
            episode.append(time_step)

        self.buffer.extend(episode)
        self._update_index(episode)

    def _update_index(self, episode: List) -> None:
        eps_len = len(episode)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

        self._index.append({
            "length": eps_len,
            "timestamp": timestamp,
            "buffer_index": len(self.buffer) - eps_len
        })

        data_path = ROOT_PATH / "data" / "replay_buffer" / self.config['dataset']['name']
        write_json(self._index, str(data_path / "index.json"))

    def __getitem__(self, ind: int) -> Dict[str, torch.Tensor]:
        episode_start = 0
        for episode_info in self._index:
            if ind < episode_start + episode_info["length"]:
                break
            episode_start += episode_info["length"]

        time_step = self.buffer[ind]

        reward = np.zeros_like(time_step.reward)
        discount = np.ones_like(time_step.discount)

        for i in range(self.nstep):
            if ind + i >= len(self.buffer):
                break
            next_step = self.buffer[ind + i]
            reward += discount * next_step.reward
            discount *= next_step.discount * self.discount

        instance_data = {
            "observation": torch.from_numpy(time_step.observation).float(),
            "action": torch.from_numpy(time_step.action).float(),
            "reward": torch.from_numpy(reward).float(),
            "discount": torch.from_numpy(discount).float(),
            "next_observation": torch.from_numpy(self.buffer[ind + self.nstep - 1].observation).float()
        }

        return self.preprocess_data(instance_data)