import os, sys
import numpy as np
import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


class LeRobotEMMADataset(LeRobotDataset):
    def __init__(self, data_id, image_key='image', sampling_rate=10, name="lerobot_dataset", **kwargs):
        super().__init__(data_id, episodes=[0, 1, 2, 3, 4])

        self.paths = []
        self.frame_names = []
        self.trajectories = []
        self.tasks = []
        self.actions = []

        self.sampling_rate = sampling_rate

        for i, episode_idx in enumerate(self.episodes):
            self.paths.append(self.meta.get_data_file_path(episode_idx))

            start_idx = self.episode_data_index['from'][i].item()
            end_idx = self.episode_data_index['to'][i].item()
            frames = []
            frame_names = []
            tasks = []
            actions = []

            delta_action = None
            for t in range(start_idx, end_idx):
                data = super().__getitem__(t)
                action = data['action']
                action = action.numpy().astype(np.float32)
                if delta_action is None:
                    delta_action = action[:-1]
                else:
                    delta_action += action[:-1]
                
                if t % self.sampling_rate == 0:
                    task = data['task']
                    image = data[f'observation.images.{image_key}'] * 255.0
                    image = image.numpy().astype(np.uint8).transpose(1, 2, 0)
                    frames.append(image)
                    frame_names.append(f"episode_{episode_idx}_frame_{t}.jpg")
                    tasks.append(task)
                    action = np.concatenate([delta_action, action[-1:]])
                    actions.append(action)
            self.tasks.append(tasks)
            self.actions.append(np.array(actions))
            self.trajectories.append(
                np.array(frames)
            )
            self.frame_names.append(frame_names)

        self.name = name

    def __getitem__(self, idx):        
        frames = self.trajectories[idx]
        
        data = {}
        data['task'] = self.tasks[idx]
        data["images"] = frames
        data["path"] = np.array([self.paths[idx]] * len(frames))
        data["frame_names"] = np.array(self.frame_names[idx])
        data["action"] = self.actions[idx]

        return data

    def __len__(self):
        return len(self.episodes)
