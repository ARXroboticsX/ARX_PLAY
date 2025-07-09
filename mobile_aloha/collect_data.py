# -- coding: UTF-8
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

import time
import h5py
import argparse
import rospy
import cv2
import yaml
import threading
import pyttsx3

import numpy as np

from copy import deepcopy

from utils.ros_operator import RosOperator

np.set_printoptions(linewidth=200)

voice_engine = pyttsx3.init()
voice_engine.setProperty('voice', 'en')
voice_engine.setProperty('rate', 120)  # 设置语速


def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File not found - {yaml_file}")

        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file - {e}")

        return None


def voice_process(voice_engine, line):
    voice_engine.say(line)
    voice_engine.runAndWait()
    print(line)

    return


# 根据动作和编码范围更新ready_flag
def update_ready_flag(ready_flag, action, gripper_idx, encode_ranges):
    for idx in gripper_idx:
        if ready_flag % 2 == 1 and action[idx] < encode_ranges['close']:
            ready_flag += 1

            print(f'{ready_flag=}: {action[idx]=}')
        elif ready_flag % 2 == 0 and encode_ranges['middle'] < action[idx] < encode_ranges['max']:
            ready_flag += 1

            print(f'{ready_flag=}: {action[idx]=}')

    return ready_flag


def collect_detect(start_episode, voice_engine, ros_operator):
    init_flag = True

    rate = rospy.Rate(args.frame_rate)
    print(f"Preparing to record episode {start_episode}")

    # 倒计时
    for i in range(3, -1, -1):
        print(f"\rwaiting {i} to start recording", end='')
        rospy.sleep(0.3)

    print(f"\nStart recording program...")

    # 键盘触发录制
    if args.key_collect:
        input("Enter any key to record :")
    else:
        if init_flag:
            init_done = False

            while not init_done and not rospy.is_shutdown():
                obs_dict = ros_operator.get_observation()
                if obs_dict == None:
                    print("synchronization frame")
                    rate.sleep()

                    continue
                action = obs_dict['eef']  # qpos

                # 减少不必要的循环
                init_done = all(val <= 0.05 for val in action)

                if init_done:
                    voice_process(voice_engine, f"init for {start_episode % 100}")
                rate.sleep()

            init_flag = False

        # 机械臂准备阶段
        ready_flag = 0
        gripper_idx = [6, 13]
        encode_ranges = {
            'close': 0.1,
            'middle': 1.0,
            'max': 5.0
        }

        while ready_flag < 2 and not rospy.is_shutdown():
            obs_dict = ros_operator.get_observation()
            if obs_dict == None:
                print("synchronization frame")
                rate.sleep()

                continue

            action = obs_dict['qpos']
            ready_flag = update_ready_flag(ready_flag, action, gripper_idx, encode_ranges)

            if ready_flag == 2:
                voice_process(voice_engine, "go")

            rate.sleep()

        return True


def collect_information(args, ros_operator, voice_engine):
    timesteps = []
    actions = []
    actions_eef = []
    action_bases = []
    action_velocities = []
    count = 0
    rate = rospy.Rate(args.frame_rate)

    # 初始化机器人基础位置
    ros_operator.init_robot_base_pose()

    gripper_idx = [6, 13]
    gripper_close = 3.0

    while (count < args.max_timesteps) and not rospy.is_shutdown():
        obs_dict = ros_operator.get_observation(ts=count)
        action_dict = ros_operator.get_action()

        # 同步帧检测
        if obs_dict is None or action_dict is None:
            print("Synchronization frame")
            rate.sleep()

            continue

        # 获取动作和观察值
        action = deepcopy(obs_dict['qpos'])
        action_eef = deepcopy(obs_dict['eef'])
        action_base = obs_dict['robot_base']
        action_velocity = obs_dict['base_velocity']

        # 夹爪动作处理
        for idx in gripper_idx:
            action[idx] = 0 if action[idx] < gripper_close else action[idx]
        action_eef[6] = 0 if action_eef[6] < gripper_close else action_eef[6]
        action_eef[13] = 0 if action_eef[13] < gripper_close else action_eef[13]

        # 检查是否超过100帧，并判断是否应该停止
        if count > 100:
            if all(val <= 0.03 for val in action):
                voice_process(voice_engine, "Back to init")

                break

        # 收集数据
        timesteps.append(obs_dict)
        actions.append(action)
        actions_eef.append(action_eef)
        action_bases.append(action_base)
        action_velocities.append(action_velocity)

        count += 1
        print(f"Frame data: {count}")

        if rospy.is_shutdown():
            exit(-1)

        rate.sleep()

    print(f"\nlen(timesteps): {len(timesteps)}")
    print(f"len(actions)  : {len(actions)}")

    return timesteps, actions, actions_eef, action_bases, action_velocities


# 保存数据函数
def save_data(args, timesteps, actions, actions_eef, action_bases, action_velocities, ros_operator, dataset_path):
    data_size = len(actions)

    # 数据字典
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/observations/eef': [],
        '/observations/robot_base': [],
        '/action': [],
        '/action_eef': [],
        '/action_base': [],
        '/action_velocity': [],
    }

    # 初始化相机字典
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image:
            data_dict[f'/observations/images_depth/{cam_name}'] = []

    # 遍历并收集数据
    while actions and not rospy.is_shutdown():
        action = actions.pop(0)  # 动作  当前动作
        action_eef = actions_eef.pop(0)
        action_base = action_bases.pop(0)
        action_velocity = action_velocities.pop(0)
        ts = timesteps.pop(0)  # 奖励  前一帧

        # 填充数据
        data_dict['/observations/qpos'].append(ts['qpos'])
        data_dict['/observations/qvel'].append(ts['qvel'])
        data_dict['/observations/eef'].append(ts['eef'])
        data_dict['/observations/effort'].append(ts['effort'])
        data_dict['/observations/robot_base'].append(ts['robot_base'])
        data_dict['/action'].append(action)
        data_dict['/action_eef'].append(action_eef)
        data_dict['/action_base'].append(action_base)
        data_dict['/action_velocity'].append(action_velocity)

        # 相机数据
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts['images'][cam_name])
            if args.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts['images_depth'][cam_name])

    # 压缩图像数据
    if args.is_compress:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # 压缩质量
        compressed_len = []
        for cam_name in args.camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_len.append([])  # 压缩的长度

            for image in image_list:
                result, encoded_image = cv2.imencode('.jpg', image, encode_param)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))

            # 更新图像
            data_dict[f'/observations/images/{cam_name}'] = compressed_list

        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()  # 取最大的图像长度，图像压缩后就是一个buf序列

        for cam_name in args.camera_names:
            # if cam_name == 'cam_binocular':
            #     continue
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)

            # 更新压缩后的图像列表
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list

        if args.use_depth_image:
            compressed_len_depth = []

            for cam_name in args.camera_names:
                depth_list = data_dict[f'/observations/images_depth/{cam_name}']
                compressed_list_depth = []
                compressed_len_depth.append([])  # 压缩的长度

                for depth in depth_list:
                    result, encoded_depth = cv2.imencode('.jpg', depth, encode_param)
                    compressed_list_depth.append(encoded_depth)
                    compressed_len_depth[-1].append(len(encoded_depth))

                # 更新图像
                data_dict[f'/observations/images_depth/{cam_name}'] = compressed_list_depth

            compressed_len_depth = np.array(compressed_len_depth)
            padded_size_depth = compressed_len_depth.max()

            for cam_name in args.camera_names:
                compressed_depth_list = data_dict[f'/observations/images_depth/{cam_name}']
                padded_compressed_depth_list = []
                for compressed_depth in compressed_depth_list:
                    padded_compressed_depth = np.zeros(padded_size_depth, dtype='uint8')
                    depth_len = len(compressed_depth)
                    padded_compressed_depth[:depth_len] = compressed_depth
                    padded_compressed_depth_list.append(padded_compressed_depth)
                data_dict[f'/observations/images_depth/{cam_name}'] = padded_compressed_depth_list

    # 文本的属性：
    # 1 是否仿真
    # 2 图像是否压缩
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = True if args.is_compress else False

        obs_dict = root.create_group('observations')
        image = obs_dict.create_group('images')
        if args.use_depth_image:
            depth = obs_dict.create_group('images_depth')

        for cam_name in args.camera_names:
            if args.is_compress:
                image_shape = (data_size, padded_size)
                image_chunks = (1, padded_size)

                if args.use_depth_image:
                    depth_shape = (data_size, padded_size_depth)
                    depth_chunks = (1, padded_size_depth)
            else:
                image_shape = (data_size, 480, 640, 3)
                image_chunks = (1, 480, 640, 3)

                if args.use_depth_image:
                    depth_shape = (data_size, 480, 640)
                    depth_chunks = (1, 480, 640)

            _ = image.create_dataset(cam_name, image_shape, 'uint8', chunks=image_chunks)
            if args.use_depth_image:
                _ = depth.create_dataset(cam_name, depth_shape, 'uint8', chunks=depth_chunks)

        states_dim = 14
        eef_dim = 14

        # 观测数据集
        obs_datasets = {
            'qpos': states_dim,
            'eef': eef_dim,
            'qvel': states_dim,
            'effort': states_dim,
            'robot_base': 6,
            'base_velocity': 4
        }

        # 动作数据集
        action_datasets = {
            'action': states_dim,
            'action_eef': eef_dim,
            'action_base': 6,
            'action_velocity': 4
        }

        # 创建obs_dict数据集
        for name, dim in obs_datasets.items():
            _ = obs_dict.create_dataset(name, (data_size, dim))

        # 创建root数据集
        for name, dim in action_datasets.items():
            _ = root.create_dataset(name, (data_size, dim))

        # 将数据写入 HDF5 文件
        for name, array in data_dict.items():
            # print(f"{name=}, ", end='')
            # print(f'{len(array)=}, {len(array[0])=}')

            root[name][...] = array

    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n' % dataset_path)

    return


def main(args):
    config = load_yaml(args.config)
    ros_operator = RosOperator(args, config, in_collect=True)

    datasets_dir = args.datasets if sys.stdin.isatty() else Path.joinpath(ROOT, args.datasets)

    num_episodes = 100 if args.episode_idx == -1 else 1
    current_episode = 0 if args.episode_idx == -1 else args.episode_idx

    episode_num = 0
    # 检测已存在的
    for ex_idx in range(current_episode, 1000):
        episode_search_path = os.path.join(datasets_dir, f'episode_{ex_idx}.hdf5')  # save path
        if os.path.exists(episode_search_path) == True:
            current_episode = current_episode + 1
        else:
            break

    while episode_num < num_episodes and not rospy.is_shutdown():
        print(f'Episode {episode_num}')
        collect_detect(current_episode, voice_engine, ros_operator)

        print(f"Start to record episode {current_episode}")
        timesteps, actions, actions_eef, action_bases, action_velocities = collect_information(args, ros_operator,
                                                                                               voice_engine)

        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)

        dataset_path = os.path.join(datasets_dir, "episode_" + str(current_episode))
        threading.Thread(target=save_data, args=(args, timesteps, actions, actions_eef, action_bases, action_velocities,
                                                 ros_operator, dataset_path,)).start()
        # voice_process(voice_engine, f"save done")
        episode_num = episode_num + 1
        current_episode = current_episode + 1


def parse_arguments(known=False):
    parser = argparse.ArgumentParser()

    # 数据集配置
    parser.add_argument('--datasets', type=str, default=Path.joinpath(ROOT, 'datasets'),
                        help='dataset dir')
    parser.add_argument('--episode_idx', type=int, default=0, help='episode index')
    parser.add_argument('--max_timesteps', type=int, default=800, help='max timesteps')
    parser.add_argument('--frame_rate', type=int, default=60, help='frame rate')

    # 配置文件
    parser.add_argument('--config', type=str,
                        default=Path.joinpath(ROOT, 'data/config.yaml'),
                        help='config file')

    # 图像处理选项
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['head', 'left_wrist', 'right_wrist'],
                        default=['head', 'left_wrist', 'right_wrist'], help='camera names')
    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')
    parser.add_argument('--is_compress', action='store_true', help='compress image')

    # 机器人选项
    parser.add_argument('--use_base', action='store_true', help='use robot base')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance',
                        help='record data')

    # 数据采集选项
    parser.add_argument('--key_collect', action='store_true', help='use key collect')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
