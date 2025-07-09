import numpy as np
import torch
import os
import h5py
import json
from torch.utils.data import TensorDataset, DataLoader
import random
import IPython

e = IPython.embed
import cv2
from scipy.spatial.transform import Rotation as R  # eef:ZXY

FILTER_MISTAKES = False  # Filter out mistakes from the dataset even if not use_language


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, policy_config, norm_stats, arm_delay_time):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids  # 1000
        self.dataset_dir = dataset_dir

        self.chunk_size = policy_config['chunk_size']

        self.norm_stats = norm_stats

        self.is_sim = None

        self.camera_names = policy_config['camera_names']

        self.use_base = policy_config['use_base']

        self.use_depth_image = policy_config['use_depth_image']

        self.arm_delay_time = arm_delay_time

        if policy_config['policy_class'] == "ACT":
            self.use_qvel = policy_config['use_qvel']
            self.use_effort = policy_config['use_effort']

            self.command_list = [cmd.strip("'\"") for cmd in policy_config['command_list']]

            self.add_action_output = True
        else:
            self.use_qvel = False
            self.use_effort = False

            self.add_action_output = False

        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # if val datasets True

        episode_id = self.episode_ids[index]

        # 读取数据
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            is_compress = root.attrs['compress']
            actions = root['action']

            if self.use_base:
                actions = np.concatenate((actions, np.array(root['action_base'])), axis=1)
                actions = np.concatenate((actions, np.array(root['action_velocity'])), axis=1)

            original_action_shape = actions.shape  # [:,:7]
            max_action_len = original_action_shape[0]  # max_episode
            start_ts = np.random.choice(max_action_len)  # 随机抽取一个索引

            states2action_step = 1
            actions = actions[states2action_step:]  # 错开了一帧 # ,
            last_action = actions[-1]
            last_action = np.tile(last_action[np.newaxis, :], (states2action_step, 1))
            actions = np.append(actions, last_action, axis=0)  # actions[-1][np.newaxis, :]

            if self.add_action_output:
                action_zero_addition = np.zeros(original_action_shape)
                actions = np.concatenate((actions, action_zero_addition),
                                         axis=1)  # 14 -> 28 # robot base 19 -> 38
            additional_action_shape = actions.shape

            qpos = root['/observations/qpos'][start_ts]
            eef = root['/observations/eef'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            effort = root['/observations/effort'][start_ts]
            robot_base = root['/observations/robot_base'][start_ts, :3]  # 9
            robot_head = root['/observations/robot_base'][start_ts, 3:6]  # 9
            base_velocity = root['/observations/base_velocity'][start_ts]

            states_init = root['/observations/eef'][0]

            joints_dim = 7

            left_states_init = states_init[:joints_dim]
            left_qpos = qpos[:joints_dim]
            left_states = left_qpos

            left_states = np.concatenate((left_states, qvel[:joints_dim]),
                                         axis=0) if self.use_qvel else left_states
            left_states = np.concatenate((left_states, effort[joints_dim - 1:joints_dim]),
                                         axis=0) if self.use_effort else left_states

            right_states_init = states_init[joints_dim:joints_dim * 2]
            right_qpos = qpos[joints_dim:joints_dim * 2]
            right_states = right_qpos

            right_states = np.concatenate((right_states, qvel[joints_dim:joints_dim * 2]),
                                          axis=0) if self.use_qvel else right_states
            right_states = np.concatenate((right_states, effort[joints_dim * 2 - 1:joints_dim * 2]),
                                          axis=0) if self.use_effort else right_states

            left_states = np.concatenate((left_states, right_states), axis=0)

            right_states = left_states

            image_dict = dict()
            image_depth_dict = dict()
            for cam_name in self.camera_names:
                if is_compress:
                    decoded_image = root[f'/observations/images/{cam_name}'][start_ts]
                    image_dict[cam_name] = cv2.imdecode(decoded_image, 1)
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

                if self.use_depth_image:
                    if is_compress:
                        decoded_image = root[f'/observations/images/{cam_name}'][start_ts]
                        image_depth_dict[cam_name] = cv2.imdecode(decoded_image, 1)
                    else:
                        image_depth_dict[cam_name] = root[f'/observations/images_depth/{cam_name}'][start_ts]

            start_action = min(start_ts, max_action_len - 1)

            index = max(0, start_action - self.arm_delay_time)
            action = actions[index:]  # hack, to make timesteps more aligned

            # if self.use_robot_base:
            #     action = np.concatenate((action, root['/action_base'][index:]), axis=1)
            action_len = max_action_len - index  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(additional_action_shape, dtype=np.float32)
        # print(f'$$$$$$$$$$$$$$$${action.shape=}')
        padded_action[:action_len] = action
        is_pad_action = np.zeros(max_action_len)
        is_pad_action[action_len:] = 1
        padded_action = padded_action[:self.chunk_size]
        is_pad_action = is_pad_action[:self.chunk_size]

        # rgb图像
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data)  # Adjusting channel
        image_data = image_data / 255.0  # normalize image and change dtype to float

        # 深度图像
        image_depth_data = np.zeros(1, dtype=np.float32)
        if self.use_depth_image:
            all_cam_images_depth = []
            for cam_name in self.camera_names:
                all_cam_images_depth.append(image_depth_dict[cam_name])
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)
            # construct observations
            image_depth_data = torch.from_numpy(all_cam_images_depth)
            # image_depth_data = torch.einsum('k h w c -> k c h w', image_depth_data)
            image_depth_data = image_depth_data / 255.0

        # return
        left_states_data = torch.from_numpy(left_states).float()
        right_states_data = torch.from_numpy(right_states).float()

        action_data = torch.from_numpy(padded_action).float()
        is_pad_action = torch.from_numpy(is_pad_action).bool()

        left_states_data = ((left_states_data - self.norm_stats["left_states_mean"]) /
                            self.norm_stats["left_states_std"])
        right_states_data = ((right_states_data - self.norm_stats["right_states_mean"]) /
                             self.norm_stats["right_states_std"])
        # robot_base_data = (robot_base_data - self.norm_stats["robot_head_mean"]) / self.norm_stats["robot_head_std"]

        robot_base_data = torch.from_numpy(robot_base).float()
        robot_base_data = (robot_base_data - self.norm_stats["robot_base_mean"]) / self.norm_stats["robot_base_std"]

        robot_head_data = torch.from_numpy(robot_head).float()
        robot_head_data = (robot_head_data - self.norm_stats["robot_head_mean"]) / self.norm_stats["robot_head_std"]

        base_velocity_data = torch.from_numpy(base_velocity).float()
        base_velocity_data = (base_velocity_data - self.norm_stats["base_velocity_mean"]) / self.norm_stats[
            "base_velocity_std"]

        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        action_data = torch.tensor(action_data, dtype=torch.float)

        return (image_data, image_depth_data, left_states_data, right_states_data, robot_base_data, robot_head_data,
                base_velocity_data, action_data, is_pad_action)


def get_IO_for_norm(qpos, eef, qvel, effort, action, policy_config):
    if policy_config['policy_class'] == "ACT":
        use_qvel = policy_config['use_qvel']
        use_effort = policy_config['use_effort']
        add_action_output = True
    else:
        use_qvel = False
        use_effort = False
        add_action_output = False

    joints_dim = 7

    # left or single
    left_qpos = qpos[:, :joints_dim]
    left_states = left_qpos

    # right
    right_qpos = qpos[:, joints_dim:joints_dim * 2]
    right_states = right_qpos

    left_states = np.concatenate((left_states, qvel[:, :joints_dim]),
                                 axis=1) if use_qvel else left_states
    left_states = np.concatenate((left_states, effort[:, joints_dim - 1:joints_dim]),
                                 axis=1) if use_effort else left_states

    right_states = np.concatenate((right_states, qvel[:, joints_dim:joints_dim * 2]),
                                  axis=1) if use_qvel else right_states
    right_states = np.concatenate((right_states, effort[:, joints_dim * 2 - 1:joints_dim * 2]),
                                  axis=1) if use_effort else right_states

    left_states = np.concatenate((left_states, right_states), axis=1)

    right_states = left_states

    if add_action_output:
        action_zero_addition = np.zeros(action.shape)
        action = np.concatenate((action, action_zero_addition), axis=1)  # 14 -> 28 or 7 -> 14

    return left_states, right_states, action


def get_norm_stats(dataset_dir, num_episodes, policy_config):
    all_left_states_data = []
    all_right_states_data = []
    all_action_data = []
    all_robot_head_data = []
    all_robot_base_data = []
    all_robot_velocity_data = []

    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')

        try:
            with h5py.File(dataset_path, 'r') as root:
                try:
                    qpos = root['/observations/qpos'][()]
                    eef = root['/observations/eef'][()]
                    qvel = root['/observations/qvel'][()]
                    effort = root['/observations/effort'][()]
                    robot_base = root['/observations/robot_base'][()]
                    base_velocity = root['/observations/base_velocity'][()]
                    action = root['/action'][()]
                except KeyError as e:
                    print(f"Key error in file {dataset_path}: {e}")
                except ValueError as e:
                    print(f"Value error while processing file {dataset_path}: {e}")

                if policy_config['use_base']:
                    action = np.concatenate((action, root['/action_base'][()]), axis=1)
                    action = np.concatenate((action, root['/action_velocity'][()]), axis=1)
        except FileNotFoundError:
            print(f"File not found: {dataset_path}")
        except OSError as e:
            print(f"OS error when accessing file {dataset_path}: {e}")

        left_states, right_states, action = get_IO_for_norm(qpos, eef, qvel, effort, action, policy_config)

        all_left_states_data.append(torch.from_numpy(left_states))
        all_right_states_data.append(torch.from_numpy(right_states))
        all_action_data.append(torch.from_numpy(action))
        all_robot_base_data.append(torch.from_numpy(robot_base[:, :3]))
        all_robot_head_data.append(torch.from_numpy(robot_base[:, 3:6]))
        all_robot_velocity_data.append(torch.from_numpy(base_velocity))

    # 以最少的为准，多的就才减掉后面的
    episode_len_min = min(arr.shape[0] for arr in all_left_states_data)
    episode_len_max = max(arr.shape[0] for arr in all_left_states_data)
    target_demo_len = episode_len_max

    # print(f'{episode_len_min=}, {episode_len_max=}, {target_demo_len=}')
    for idx in range(len(all_left_states_data)):
        if (all_left_states_data[idx].shape[0] < target_demo_len):

            pad_left_states = torch.zeros((target_demo_len, all_left_states_data[idx].shape[1]))
            pad_left_states[:all_left_states_data[idx].shape[0]] = all_left_states_data[idx]
            all_left_states_data[idx] = pad_left_states

            pad_right_states = torch.zeros((target_demo_len, all_right_states_data[idx].shape[1]))
            pad_right_states[:all_right_states_data[idx].shape[0]] = all_right_states_data[idx]
            all_right_states_data[idx] = pad_right_states

            pad_action = torch.zeros((target_demo_len, all_action_data[idx].shape[1]))
            pad_action[:all_action_data[idx].shape[0]] = all_action_data[idx]
            all_action_data[idx] = pad_action

            pad_action_base = torch.zeros((target_demo_len, all_robot_base_data[idx].shape[1]))
            pad_action_base[:all_robot_base_data[idx].shape[0]] = all_robot_base_data[idx]
            all_robot_base_data[idx] = pad_action_base

            pad_action_head = torch.zeros((target_demo_len, all_robot_head_data[idx].shape[1]))
            pad_action_head[:all_robot_head_data[idx].shape[0]] = all_robot_head_data[idx]
            all_robot_head_data[idx] = pad_action_head

            pad_action_velocity = torch.zeros((target_demo_len, all_robot_velocity_data[idx].shape[1]))
            pad_action_velocity[:all_robot_velocity_data[idx].shape[0]] = all_robot_velocity_data[idx]
            all_robot_velocity_data[idx] = pad_action_velocity

    all_left_states_data = torch.stack(all_left_states_data)  # (50, 600, 14)
    all_right_states_data = torch.stack(all_right_states_data)  # (50, 600, 14)
    all_robot_base_data = torch.stack(all_robot_base_data)
    all_robot_head_data = torch.stack(all_robot_head_data)
    all_robot_velocity_data = torch.stack(all_robot_velocity_data)
    all_action_data = torch.stack(all_action_data)  # (50, 600, 14)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    left_states_mean = all_left_states_data.mean(dim=[0, 1], keepdim=True)  # [1, 1, states_dim]
    left_states_std = all_left_states_data.std(dim=[0, 1], keepdim=True)  # [1, 1, states_dim]
    right_states_mean = all_right_states_data.mean(dim=[0, 1], keepdim=True)  # [1, 1, states_dim]
    right_states_std = all_right_states_data.std(dim=[0, 1], keepdim=True)  # [1, 1, states_dim]
    robot_head_mean = all_robot_head_data.mean(dim=[0, 1, 2], keepdim=True)  # [1, 1, states_dim]
    robot_head_std = all_robot_head_data.std(dim=[0, 1, 2], keepdim=True)  # [1, 1, states_dim]

    robot_base_mean = all_robot_base_data.mean(dim=[0, 1], keepdim=True)  # [1, 1, states_dim]
    robot_base_std = all_robot_base_data.std(dim=[0, 1], keepdim=True)  # [1, 1, states_dim]
    base_velocity_mean = all_robot_velocity_data.mean(dim=[0, 1], keepdim=True)
    base_velocity_std = all_robot_velocity_data.std(dim=[0, 1], keepdim=True)

    left_states_std = torch.clip(left_states_std, 1e-2, np.inf)  # clipping，
    right_states_std = torch.clip(right_states_std, 1e-2, np.inf)  # clipping，
    robot_head_std = torch.clip(robot_head_std, 1e-2, np.inf)  # clipping，
    robot_base_std = torch.clip(robot_base_std, 1e-2, np.inf)  # clipping，
    base_velocity_std = torch.clip(base_velocity_std, 1e-2, np.inf)

    stats = {"action_mean": action_mean.numpy().squeeze(),
             "action_std": action_std.numpy().squeeze(),
             "left_states_mean": left_states_mean.numpy().squeeze(),
             "left_states_std": left_states_std.numpy().squeeze(),
             "right_states_mean": right_states_mean.numpy().squeeze(),
             "right_states_std": right_states_std.numpy().squeeze(),
             "robot_base_std": robot_base_std.numpy().squeeze(),  # robot base
             "robot_base_mean": robot_base_mean.numpy().squeeze(),
             "robot_head_std": robot_head_std.numpy().squeeze(),
             "robot_head_mean": robot_head_mean.numpy().squeeze(),
             "base_velocity_std": base_velocity_std.numpy().squeeze(),
             "base_velocity_mean": base_velocity_mean.numpy().squeeze(),
             }

    return stats


def load_data(dataset_dir, num_episodes, arm_delay_time, policy_config, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')

    # obtain train test split
    train_ratio = 0.8  # 数据集比例
    shuffled_indices = np.random.permutation(num_episodes)  # 打乱

    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]  # eval all but train 80%

    # obtain normalization stats for eef and action  返回均值和方差
    norm_stats = get_norm_stats(dataset_dir, num_episodes, policy_config)

    # construct dataset and dataloader 归一化处理  结构化处理数据
    train_dataset = EpisodicDataset(train_indices, dataset_dir, policy_config, norm_stats, arm_delay_time)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, policy_config, norm_stats, arm_delay_time)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1,
                                prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


# env utils
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])

    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


# helper functions
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_gpu_mem_info(gpu_id=0):
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} not exist!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    import psutil

    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)

    return mem_total, mem_free, mem_process_used
