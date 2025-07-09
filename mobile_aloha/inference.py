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

import argparse
import collections
import pickle
import yaml
from einops import rearrange
import numpy as np
import rospy
import torch

from utils.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from utils.utils import set_seed  # helper functions

from utils.ros_operator import RosOperator
from functools import partial
import signal
import sys

obs_dict = collections.OrderedDict()

np.set_printoptions(linewidth=200)


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


def get_model_config(args):
    set_seed(args.seed)

    base_config = {
        'lr': args.lr,
        'lr_backbone': args.lr_backbone,
        'weight_decay': args.weight_decay,
        'loss_function': args.loss_function,

        'backbone': args.backbone,
        'chunk_size': args.chunk_size,
        'hidden_dim': args.hidden_dim,

        'camera_names': args.camera_names,

        'position_embedding': args.position_embedding,
        'masks': args.masks,
        'dilation': args.dilation,

        'use_base': args.use_base,

        'use_depth_image': args.use_depth_image,
    }

    if args.policy_class == 'ACT':
        policy_config = {
            **base_config,
            'enc_layers': args.enc_layers,
            'dec_layers': args.dec_layers,
            'nheads': args.nheads,
            'dropout': args.dropout,
            'pre_norm': args.pre_norm,
            'states_dim': 7,
            'action_dim': 7,
            'kl_weight': args.kl_weight,
            'dim_feedforward': args.dim_feedforward,

            'use_qvel': args.use_qvel,
            'use_effort': args.use_effort,
            'use_eef_states': args.use_eef_states,

            'command_list': [],
        }

        # 更新 states_dim
        policy_config['states_dim'] += policy_config['action_dim'] if args.use_qvel else 0
        policy_config['states_dim'] += 1 if args.use_effort else 0
        policy_config['states_dim'] *= 2

        # 更新 action_dim
        policy_config['action_dim'] *= 2  # 双臂预测
        policy_config['action_dim'] += 10 if args.use_base else 0
        policy_config['action_dim'] *= 2  # 增加预测量
        print("policy_config['action_dim']=", policy_config['action_dim'],
              "policy_config['states_dim']=", policy_config['states_dim'])

    elif args.policy_class == 'CNNMLP':
        policy_config = {
            **base_config,
            'action_dim': 14,
            'states_dim': 14,
        }
    elif args.policy_class == 'Diffusion':
        policy_config = {
            **base_config,
            'observation_horizon': args.observation_horizon,
            'action_horizon': args.action_horizon,
            'num_inference_timesteps': args.num_inference_timesteps,
            'ema_power': args.ema_power,

            'action_dim': 14,
            'states_dim': 14,
        }
    else:
        raise NotImplementedError

    config = {
        'ckpt_dir': args.ckpt_dir if sys.stdin.isatty() else Path.joinpath(ROOT, args.ckpt_dir),
        'ckpt_name': args.ckpt_name,
        'ckpt_stats_name': args.ckpt_stats_name,
        'episode_len': args.max_publish_step,
        'state_dim': policy_config['states_dim'],
        'policy_class': args.policy_class,
        'policy_config': policy_config,
        'temporal_agg': args.temporal_agg,
        'camera_names': args.camera_names,
    }

    return config


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError

    return policy


def get_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        # print(f'{cam_name=}')
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)

    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    return curr_image


def get_depth_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_images.append(observation['images_depth'][cam_name])
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    return curr_image


def apply_gripper_gate(action_value, gate):
    return 0 if action_value < gate else action_value


def robot_action(ros_operator, args, action):
    gripper_gate = args.gripper_gate
    max_gripper = 5

    gripper_idx = [6, 13]

    left_action = action[:gripper_idx[0] + 1]  # 取8维度
    if gripper_gate != -1:
        left_action[gripper_idx[0]] = apply_gripper_gate(left_action[gripper_idx[0]], gripper_gate)

    right_action = action[gripper_idx[0] + 1:gripper_idx[1] + 1]
    if gripper_gate != -1:
        right_action[gripper_idx[0]] = apply_gripper_gate(left_action[gripper_idx[0]], gripper_gate)

    ros_operator.follow_arm_publish(left_action, right_action)

    if args.use_base:
        action_base = action[gripper_idx[1] + 1:gripper_idx[1] + 1 + 10]
        ros_operator.set_robot_base_target(action_base)


def model_inference(args, config, ros_operator, policy):
    global obs_dict

    # 加载数据集统计信息
    stats_path = os.path.join(config['ckpt_dir'], config['ckpt_stats_name'])
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    chunk_size = config['policy_config']['chunk_size']
    hidden_dim = config['policy_config']['hidden_dim']
    action_dim = config['policy_config']['action_dim']

    use_qvel = config['policy_config']['use_qvel']
    use_effort = config['policy_config']['use_effort']
    use_eef_states = config['policy_config']['use_eef_states']

    # use_robot_base = config['policy_config']['use_robot_base']
    action = np.zeros((action_dim,))

    pre_left_states_process = lambda s_left_states: ((s_left_states - stats['left_states_mean']) /
                                                     stats['left_states_std'])
    pre_right_states_process = lambda s_right_states: ((s_right_states - stats['right_states_mean']) /
                                                       stats['right_states_std'])
    pre_robot_base_process = lambda s_robot_base: (s_robot_base - stats['robot_base_mean']) / stats['robot_base_std']
    pre_robot_head_process = lambda s_robot_head: (s_robot_head - stats['robot_head_mean']) / stats['robot_head_std']
    pre_base_velocity_process = lambda s_base_velocity: (s_base_velocity - stats['base_velocity_mean']) / stats[
        'base_velocity_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']  # 正确对应的

    left_states_init = [0, 0, 0, 0, 0, 0, 0]
    right_states_init = [0, 0, 0, 0, 0, 0, 0]

    policy.cuda()
    policy.eval()

    max_publish_step = config['episode_len']

    while not rospy.is_shutdown():  # 由于限制，max_publish_step最大步数不能太大，显存或内存承受不住，所以外套一个循环
        if config['temporal_agg']:
            print(f"{config['state_dim']=}")

            all_time_actions = np.zeros(
                [max_publish_step, max_publish_step + chunk_size, action_dim])  # (10000 , 10100, 14)

        timestep = 0
        skip_t = 0

        with torch.inference_mode():
            while timestep < args.max_publish_step and not rospy.is_shutdown():
                obs_dict = get_obervations(args, timestep, ros_operator)

                gripper_idx = [6, 13]

                left_qpos = obs_dict['gpos'][:gripper_idx[0] + 1] if use_eef_states \
                    else obs_dict['qpos'][:gripper_idx[0] + 1]
                left_states = left_qpos

                right_qpos = obs_dict['gpos'][gripper_idx[0] + 1:gripper_idx[1] + 1] if use_eef_states \
                    else obs_dict['qpos'][gripper_idx[0] + 1:gripper_idx[1] + 1]
                right_states = right_qpos

                left_states = np.concatenate((left_states, right_states), axis=0)

                right_states = left_states

                robot_base = obs_dict['robot_base'][:3]
                robot_base = pre_robot_base_process(robot_base)
                robot_base = torch.from_numpy(robot_base).float().cuda().unsqueeze(0)

                robot_head = obs_dict['robot_base'][3:6]
                robot_head = pre_robot_head_process(robot_head)
                robot_head = torch.from_numpy(robot_head).float().cuda().unsqueeze(0)

                base_velocity = obs_dict['base_velocity']
                base_velocity = pre_base_velocity_process(base_velocity)
                base_velocity = torch.from_numpy(base_velocity).float().cuda().unsqueeze(0)

                left_states = pre_left_states_process(left_states)
                left_states = torch.from_numpy(left_states).float().cuda().unsqueeze(0)

                right_states = pre_right_states_process(right_states)
                right_states = torch.from_numpy(right_states).float().cuda().unsqueeze(0)

                curr_image = get_image(obs_dict, config['camera_names'])
                curr_depth_image = None

                if args.use_depth_image:
                    curr_depth_image = get_depth_image(obs_dict, config['camera_names'])

                if config['policy_class'] == "ACT":
                    all_actions, image_feature = policy(curr_image, curr_depth_image, left_states, right_states,
                                                        robot_base=robot_base, robot_head=robot_head,
                                                        base_velocity=base_velocity,)
                    if config['temporal_agg']:
                        all_time_actions[[timestep], timestep:timestep + chunk_size] = all_actions.cpu().numpy()

                        actions_for_curr_step = all_time_actions[:, timestep]  # (10000,1,14) => (10000, 14)
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]

                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        if args.pos_lookahead_step != 0:
                            raw_action = all_actions[:, timestep % args.pos_lookahead_step]
                        else:
                            raw_action = all_actions[:, timestep % chunk_size]
                else:
                    raise NotImplementedError

                action = post_process(raw_action[0])

                robot_action(ros_operator, args, action)

                timestep += 1

            if args.use_base:
                action[16] = 0
                action[17] = 0
                action[19] = 0

            robot_action(ros_operator, args, action)


def get_obervations(args, timestep, ros_operator):
    global obs_dict

    rate = rospy.Rate(args.frame_rate)
    while True and not rospy.is_shutdown():
        obs_dict = ros_operator.get_observation(ts=timestep)
        if not obs_dict:
            print("syn fail")
            rate.sleep()

            continue

        return obs_dict


def init_robot(ros_operator, use_base):
    init0 = [0, 0, 0, 0, 0, 0, 4]
    init1 = [0, 0, 0, 0, 0, 0, 0]

    # 发布初始位置（关节空间姿态）
    ros_operator.follow_arm_publish_continuous(init0, init0)
    # ros_operator.robot_base_shutdown()
    input("Enter any key to continue :")

    ros_operator.follow_arm_publish_continuous(init1, init1)
    if use_base:
        ros_operator.start_base_control_thread()


def signal_handler(signal, frame, ros_operator):
    print('Caught Ctrl+C / SIGINT signal')

    # 底盘给零
    ros_operator.base_enable = False
    ros_operator.robot_base_shutdown()
    ros_operator.base_control_thread.join()

    sys.exit(0)


def parse_args(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_publish_step', type=int, default=10000, help='max publish step')

    # 数据集和检查点设置
    parser.add_argument('--ckpt_dir', type=str, default=Path.joinpath(ROOT, 'weights'),
                        help='ckpt dir')
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt',
                        help='ckpt name')
    parser.add_argument('--pretrain_ckpt', type=str, default='',
                        help='pretrain ckpt')
    parser.add_argument('--ckpt_stats_name', type=str, default='dataset_stats.pkl',
                        help='ckpt stats name')

    # 配置文件
    parser.add_argument('--data', type=str,
                        default=Path.joinpath(ROOT, 'data/config.yaml'),
                        help='config file')

    # 推理设置
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay rate')
    parser.add_argument('--loss_function', type=str, choices=['l1', 'l2', 'l1+l2'],
                        default='l1', help='loss function')
    parser.add_argument('--pos_lookahead_step', type=int, default=0, help='position lookahead step')

    # 模型结构设置
    parser.add_argument('--policy_class', type=str, choices=['CNNMLP', 'ACT', 'Diffusion'], default='ACT',
                        help='policy class selection')
    parser.add_argument('--backbone', type=str, default='resnet18', help='backbone model architecture')
    parser.add_argument('--chunk_size', type=int, default=30, help='chunk size for input data')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden layer dimension size')

    # 摄像头和位置嵌入设置
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['head', 'left_wrist', 'right_wrist'],
                        default=['head', 'left_wrist', 'right_wrist'],
                        help='camera names to use')
    parser.add_argument('--position_embedding', type=str, choices=('sine', 'learned'), default='sine',
                        help='type of positional embedding to use')
    parser.add_argument('--masks', action='store_true', help='train segmentation head if provided')
    parser.add_argument('--dilation', action='store_true',
                        help='replace stride with dilation in the last convolutional block (DC5)')

    # 机器人设置
    parser.add_argument('--use_base', action='store_true', help='use robot base')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance',
                        help='record data')
    parser.add_argument('--frame_rate', type=int, default=60, help='frame rate')

    # ACT模型专用设置
    parser.add_argument('--enc_layers', type=int, default=4, help='number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=7, help='number of decoder layers')
    parser.add_argument('--nheads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate in transformer layers')
    parser.add_argument('--pre_norm', action='store_true', help='use pre-normalization in transformer')
    parser.add_argument('--states_dim', type=int, default=14, help='state dimension size')
    parser.add_argument('--kl_weight', type=int, default=10, help='KL divergence weight')
    parser.add_argument('--dim_feedforward', type=int, default=3200, help='feedforward network dimension')
    parser.add_argument('--temporal_agg', type=bool, default=True, help='use temporal aggregation')

    # Diffusion模型专用设置
    parser.add_argument('--observation_horizon', type=int, default=1, help='observation horizon length')
    parser.add_argument('--action_horizon', type=int, default=8, help='action horizon length')
    parser.add_argument('--num_inference_timesteps', type=int, default=10,
                        help='number of inference timesteps')
    parser.add_argument('--ema_power', type=int, default=0.75, help='EMA power for diffusion process')

    # 图像设置
    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')
    parser.add_argument('--is_compress', action='store_true', help='compress image')

    # 状态和动作设置
    parser.add_argument('--use_qvel', action='store_true', help='include qvel in state information')
    parser.add_argument('--use_effort', action='store_true', help='include effort data in state')
    parser.add_argument('--use_eef_states', action='store_true', help='use eef data in state')

    parser.add_argument('--gripper_gate', type=float, default=-1, help='gripper gate threshold')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(args):
    data = load_yaml(args.data)
    config = get_model_config(args)
    ros_operator = RosOperator(args, data, in_collect=False)

    if args.use_base:
        signal.signal(signal.SIGINT, partial(signal_handler, ros_operator=ros_operator))

    policy = make_policy(config['policy_class'], config['policy_config'])
    ckpt_dir = config['ckpt_dir'] if sys.stdin.isatty() else Path.joinpath(ROOT, config['ckpt_dir'])
    ckpt_path = os.path.join(ckpt_dir, config['ckpt_name'])
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)

    init_robot(ros_operator, args.use_base)

    model_inference(args, config, ros_operator, policy)


if __name__ == '__main__':
    args = parse_args()
    main(args)
