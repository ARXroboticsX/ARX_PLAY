# coding=utf-8
import os
import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt

DT = 0.01
# JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
JOINT_NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5"]
STATE_NAMES = JOINT_NAMES + ["gripper"]
BASE_STATE_NAMES = ["linear_vel", "angular_vel"]

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
        if 'effort' in root.keys():
            effort = root['/observations/effort'][()]
        else:
            effort = None
        action = root['/action'][()]
        base_action = root['/base_action'][()]
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
                image_list.append(image)
            image_dict[cam_name] = image_list

    return qpos, qvel, effort, action, base_action, image_dict


def main(args):
    dataset_dir = args['datasets']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx}'
    
    if episode_idx == -1:
        
        for idx in range(100):
            dataset_name = f'episode_{idx}'
            dataset_path = os.path.join(dataset_dir, dataset_name)
            
            qpos, qvel, effort, action, base_action, image_dict = load_hdf5(dataset_path)

            print(f"{dataset_path}hdf5 loaded!!")
            
            save_videos(image_dict, action, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
            visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
            # visualize_base(base_action, plot_path=os.path.join(dataset_dir, dataset_name + '_base_action.png'))
    else:
        qpos, qvel, effort, action, base_action, image_dict = load_hdf5(os.path.join(dataset_dir, dataset_name))

        print('hdf5 loaded!!')
        save_videos(image_dict, action, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))

        visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
        # visualize_base(base_action, plot_path=os.path.join(dataset_dir, dataset_name + '_base_action.png'))


def save_videos(video, actions, dt, video_path=None):
    cam_names = list(video.keys())
    all_cam_videos = []
    for cam_name in cam_names:
        all_cam_videos.append(video[cam_name])
    all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

    n_frames, h, w, _ = all_cam_videos.shape
    fps = int(1 / dt)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for t in range(n_frames):
        image = all_cam_videos[t]

        if is_compressed == False:
            image = image[:, :, [2, 1, 0]]  # swap B and R channel

        # cv2.imshow("images", image)
        # cv2.waitKey(30)
        # print("step_id: ", t,
        #       "\nleft: ", np.round(actions[t][:7], 3),
        #       "\nright: ", np.round(actions[t][7:], 3), "\n")
        out.write(image)
    out.release()

    print(f'Saved video to: {video_path}')


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list)  # ts, dim
    command = np.array(command_list)

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

    # # plot arm command
    # for dim_idx in range(num_dim):
    #     ax = axs[dim_idx]
    #     ax.plot(command[:, dim_idx], label=label2)
    #     ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()


def visualize_base(readings, plot_path=None):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets', type=str, help='dataset dir', required=True)
    parser.add_argument('--episode_idx', type=int, default=0, help='episode index')

    main(vars(parser.parse_args()))
