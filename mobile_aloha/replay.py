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

import yaml
import h5py
import argparse
import signal

import rospy

import numpy as np

np.set_printoptions(linewidth=200)

from functools import partial

from utils.ros_operator import RosOperator

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


def load_hdf5(dataset_path):
    dataset_path = Path.joinpath(ROOT, dataset_path)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at: {dataset_path}")

    try:
        with h5py.File(dataset_path, 'r') as root:
            qposes = root.get('/observations/qpos')
            eefs = root.get('/observations/eef')
            actions = root.get('/action')
            actions_eefs = root.get('/action_eef')
            action_base = root.get('/action_base')
            action_velocity = root.get('/action_velocity')

            # 确保所有所需的数据集都存在
            if any(item is None for item in [qposes, eefs, actions, actions_eefs,
                                             action_base, action_velocity]):
                missing_datasets = [name for name, item in zip(
                    ['/observations/qpos', '/observations/eef', '/action', '/action_eef',
                     '/action_base', '/action_velocity'],
                    [qposes, eefs, actions, actions_eefs, action_base, action_velocity]
                ) if item is None]

                raise ValueError(f"Missing datasets in HDF5 file: {', '.join(missing_datasets)}")

            return qposes[()], eefs[()], actions[()], actions_eefs[()], action_base[()], action_velocity[()]
    except Exception as e:
        raise RuntimeError(f"Error occurred while loading the HDF5 file: {e}")


def robot_action(ros_operator, args, action, action_base, actions_velocity, timestep):
    gripper_idx = [6, 13]

    left_action = action[:gripper_idx[0] + 1]  # 取7维度
    right_action = action[gripper_idx[0] + 1:gripper_idx[1] + 1]  # action[7:14]

    ros_operator.follow_arm_publish(left_action, right_action)  # follow_arm_publish_continuous_thread

    if args.use_base:
        ros_operator.set_robot_base_target(np.concatenate([action_base, actions_velocity]))


def init_robot(ros_operator, use_base):
    init0 = [0, 0, 0, 0, 0, 0, 4]
    init1 = [0, 0, 0, 0, 0, 0, 0]

    ros_operator.follow_arm_publish_continuous(init0, init0)
    ros_operator.robot_base_shutdown()

    if use_base:
        input("Enter any key to continue :")

        ros_operator.start_base_control_thread()
        ros_operator.follow_arm_publish_continuous(init1, init1)


def signal_handler(signal, frame, ros_operator):
    print('Caught Ctrl+C / SIGINT signal')

    # 底盘给零
    ros_operator.robot_base_shutdown()
    ros_operator.base_control_thread.join()

    sys.exit(0)

def main(args):
    config = load_yaml(args.data)
    ros_operator = RosOperator(args, config, in_collect=False)
    signal.signal(signal.SIGINT, partial(signal_handler, ros_operator=ros_operator))

    qpoes, eefs, actions, actions_eefs, action_base, actions_velocity = load_hdf5(args.episode_path)

    init_robot(ros_operator, args.use_base)

    if args.states_replay:
        replay_actions = actions
    else:
        replay_actions = qpoes

    rate = rospy.Rate(args.frame_rate)
    for idx in range(len(replay_actions)):
        print(f'{replay_actions=}')
        robot_action(ros_operator, args, replay_actions[idx], action_base[idx], actions_velocity[idx], idx)
        rate.sleep()

    ros_operator.base_enable = False

def parse_args(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--episode_path', type=str, help='episode_path', required=True)
    parser.add_argument('--frame_rate', type=int, default=60, help='frame rate')
    parser.add_argument('--data', type=str, default=Path.joinpath(ROOT, 'data/config.yaml'), help='config file')

    parser.add_argument('--use_base', action='store_true', help='use base')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance',
                        help='record data')

    parser.add_argument('--states_replay', action='store_true', help='use qpos replay')

    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')
    parser.add_argument('--is_compress', action='store_true', help='compress image')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
