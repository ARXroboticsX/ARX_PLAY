# coding=utf-8
import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))

import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt

DT = 0.01

JOINT_NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
STATE_NAMES = JOINT_NAMES + ["gripper"]

POSE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper", ]
BASE_STATE_NAMES = ["Dx", "Dy", "Dz", "height", "head_pitch", "head_yaw"]  # "base_chx", "base_chy", "base_chz"
VELOCITY_NAMES = ["motor1", "motor2", "motor3", "motor4"]
is_compressed = False


def load_hdf5(dataset_name):
    global is_compressed

    dataset_path = dataset_name + '.hdf5'
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        is_compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        eef = root['/observations/eef'][()]
        robot_base = root['/observations/robot_base'][()]
        base_velocicity = root['/observations/base_velocity'][()]

        if 'effort' in root.keys():
            effort = root['/observations/effort'][()]
        else:
            effort = None

        action = root['/action'][()]
        action_eef = root['/action_eef'][()]
        action_base = root['/action_base'][()]
        action_velocity = root['/action_velocity'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    if is_compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list):
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                # down the pixel
                # image = cv2.resize(image, (0, 0), fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)

                image_list.append(image)
            image_dict[cam_name] = image_list

    return eef, qpos, qvel, effort, action, action_eef, action_base, action_velocity, image_dict


def main(args):
    if sys.stdin.isatty():
        dataset_dir = os.path.abspath(args['datasets'])
    else:
        dataset_dir = Path.joinpath(ROOT, args['datasets'])

    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx}'

    if episode_idx == -1:
        # 查找并排序所有符合的文件
        hdf5_files = [
            os.path.join(args['datasets'], f)
            for f in os.listdir(args['datasets'])
            if f.startswith("episode_") and f.endswith(".hdf5")
        ]
        hdf5_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

        total_files = len(hdf5_files)

        # 过滤指定区间
        start = args['start']
        end = total_files if args['end'] == -1 else min(args['end'] + 1, total_files)

        for idx in range(start, end):
            dataset_name = f'episode_{idx}'
            dataset_path = os.path.join(dataset_dir, dataset_name)

            (eef, qpos, qvel, effort, action, action_eef,
             action_base, action_velocity, image_dict) = load_hdf5(dataset_path)

            print(f"{dataset_path}.hdf5 loaded!")

            save_videos(image_dict, action, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
            visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
            visualize_joints_vel(qvel, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qvel.png'))
            visualize_gpos(eef, action_eef, plot_path=os.path.join(dataset_dir, dataset_name + '_gpos.png'))
            visualize_robot_base(action_base, plot_path=os.path.join(dataset_dir, dataset_name + '_action_base.png'))
            visualize_base_velocity(action_velocity,
                                    plot_path=os.path.join(dataset_dir, dataset_name + '_action_velocity.png'))
    else:
        eef, qpos, qvel, effort, action, action_eef, action_base, action_velocity, image_dict = load_hdf5(
            os.path.join(dataset_dir, dataset_name))

        print('hdf5 loaded!!')

        save_videos(image_dict, action, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
        visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
        visualize_joints_vel(qvel, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qvel.png'))
        visualize_gpos(eef, action_eef, plot_path=os.path.join(dataset_dir, dataset_name + '_gpos.png'))
        visualize_robot_base(action_base, plot_path=os.path.join(dataset_dir, dataset_name + '_action_base.png'))


def save_videos(video, actions, dt, video_path=None):
    cam_names = list(video.keys())
    all_cam_videos = []

    for cam_name in cam_names:
        if is_compressed == False:
            if cam_name != 'cam_binocular_left' and cam_name != 'cam_binocular_right':
                video[cam_name] = video[cam_name][:, :, :, [2, 1, 0]]  # swap B and R channel

        all_cam_videos.append(video[cam_name])

    all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

    n_frames, h, w, _ = all_cam_videos.shape
    fps = int(1 / dt)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for t in range(n_frames):
        image = all_cam_videos[t]
        # cv2.imshow("images", image)
        # cv2.waitKey(30)
        # print("step_id: ", t,
        #       "\nleft: ", np.round(actions[t][:7], 3),
        #       "\nright: ", np.round(actions[t][7:], 3), "\n")
        out.write(image)
    out.release()

    print(f'Saved video to: {video_path}')


def visualize_joints(qpos_list, action, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list)  # ts, dim
    command = np.array(action)

    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1, color='orangered')
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()


def visualize_joints_vel(qvel_list, action, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qvel_list)  # ts, dim
    command = np.array(action)

    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1, color='orangered')
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()


def visualize_gpos(gpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(gpos_list)  # ts, dim
    command = np.array(command_list)

    num_ts, num_dim = qpos.shape

    # num_dim = 7 ####################

    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = [name + '_left' for name in POSE_NAMES] + [name + '_right' for name in POSE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1, color='orangered')
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved gpos plot to: {plot_path}')
    plt.close()


def visualize_robot_base(readings, plot_path=None):
    readings = np.array(readings)  # ts, dim
    num_ts, num_dim = readings.shape
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = BASE_STATE_NAMES
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(readings[:, dim_idx], label='raw')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(20) / 20, mode='same'), label='smoothed_20')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(10) / 10, mode='same'), label='smoothed_10')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(5) / 5, mode='same'), label='smoothed_5')
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()


def visualize_base_velocity(readings, plot_path=None):
    readings = np.array(readings)  # ts, dim
    num_ts, num_dim = readings.shape
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = VELOCITY_NAMES
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(readings[:, dim_idx], label='raw')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(20) / 20, mode='same'), label='smoothed_20')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(10) / 10, mode='same'), label='smoothed_10')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(5) / 5, mode='same'), label='smoothed_5')
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets', type=str, help='dataset dir', required=True)
    parser.add_argument('--episode_idx', type=int, default=0, help='episode index')
    parser.add_argument("--start", type=int, default=0, help="Start index in original file list")
    parser.add_argument("--end", type=int, default=-1, help="End index (inclusive), -1 means all")
    parser.add_argument("--individual_views", action='store_true', help="Plot each view individually")

    main(vars(parser.parse_args()))
