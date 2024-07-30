# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import argparse
import os
import sys
import threading
import yaml

import torch

from utils.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from utils.utils import set_seed  # helper functions

sys.path.append("./")

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def get_model_config(opt):
    if opt.policy_class == 'ACT':
        policy_config = {'lr': opt.lr,
                         'lr_backbone': opt.lr_backbone,
                         'backbone': opt.backbone,
                         'masks': opt.masks,
                         'weight_decay': opt.weight_decay,
                         'dilation': opt.dilation,
                         'position_embedding': opt.position_embedding,
                         'loss_function': opt.loss_function,
                         'chunk_size': opt.chunk_size,  # 查询
                         'camera_names': opt.camera_names,
                         'use_depth_image': opt.use_depth_image,
                         'use_robot_base': opt.use_robot_base,
                         'kl_weight': opt.kl_weight,  # kl散度权重
                         'hidden_dim': opt.hidden_dim,  # 隐藏层维度
                         'dim_feedforward': opt.dim_feedforward,
                         'enc_layers': opt.enc_layers,
                         'dec_layers': opt.dec_layers,
                         'nheads': opt.nheads,
                         'dropout': opt.dropout,
                         'pre_norm': opt.pre_norm
                         }
    elif opt.policy_class == 'CNNMLP':
        policy_config = {'lr': opt.lr,
                         'lr_backbone': opt.lr_backbone,
                         'backbone': opt.backbone,
                         'masks': opt.masks,
                         'weight_decay': opt.weight_decay,
                         'dilation': opt.dilation,
                         'position_embedding': opt.position_embedding,
                         'loss_function': opt.loss_function,
                         'chunk_size': opt.chunk_size,  # 查询
                         'camera_names': opt.camera_names,
                         'use_depth_image': opt.use_depth_image,
                         'use_robot_base': opt.use_robot_base
                         }
    elif opt.policy_class == 'Diffusion':
        policy_config = {'lr': opt.lr,
                         'lr_backbone': opt.lr_backbone,
                         'backbone': opt.backbone,
                         'masks': opt.masks,
                         'weight_decay': opt.weight_decay,
                         'dilation': opt.dilation,
                         'position_embedding': opt.position_embedding,
                         'loss_function': opt.loss_function,
                         'chunk_size': opt.chunk_size,  # 查询
                         'camera_names': opt.camera_names,
                         'use_depth_image': opt.use_depth_image,
                         'use_robot_base': opt.use_robot_base,
                         'observation_horizon': opt.observation_horizon,
                         'action_horizon': opt.action_horizon,
                         'num_inference_timesteps': opt.num_inference_timesteps,
                         'ema_power': opt.ema_power
                         }
    else:
        raise NotImplementedError

    config = {
        'ckpt_dir': opt.ckpt_dir,
        'ckpt_name': opt.ckpt_name,
        'policy_class': opt.policy_class,
        'policy_config': policy_config,
        'onnx_dir': opt.onnx_dir,
        'onnx_name': opt.onnx_name,
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


def export_model(config):
    # 创建模型数据
    policy = make_policy(config['policy_class'], config['policy_config'])
    # print("model structure\n", policy.model)

    # 加载模型权重
    ckpt_path = os.path.join(config['ckpt_dir'], config['ckpt_name'])
    state_dict = torch.load(ckpt_path)
    new_state_dict = {}

    for key, value in state_dict.items():
        if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
            continue
        if key in ["model.input_proj_next_action.weight", "model.input_proj_next_action.bias"]:
            continue
        new_state_dict[key] = value

    loading_status = policy.deserialize(new_state_dict)
    if not loading_status:
        print("ckpt path not exist")
        return False

    # 模型设置为cuda模式和验证模式
    policy.cuda()
    policy.eval()

    camera_num = len(config['policy_config'].get('camera_names'))

    image = torch.randn(1, camera_num, 3, 480, 640).cuda()
    if config['policy_config'].get('use_depth_image', True):
        depth_image = torch.randn(1, camera_num, 3, 480, 640).cuda()
    else:
        depth_image = None
    if config['policy_config'].get('use_robot_base', True):
        state = torch.randn(1, 16).cuda()
    else:
        state = torch.randn(1, 14).cuda()

    # 导出onnxx
    if not os.path.exists(config['onnx_dir']):
        os.makedirs(config['onnx_dir'])
    onnx_path = os.path.join(config['onnx_dir'], config['onnx_name'])
    torch.onnx.export(policy, (image, depth_image, state), onnx_path)
    print('Export to ONNX: {}'.format(onnx_path))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_dir', type=str, help='ckpt dir', default='./weights')
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt', help='ckpt name')

    parser.add_argument('--onnx_dir', type=str, help='onnx dir', default='./weights')
    parser.add_argument('--onnx_name', type=str, default='policy_best.onnx', help='onnx name')

    parser.add_argument('--policy_class', type=str, choices=['CNNMLP', 'ACT', 'Diffusion'],
                        default='ACT', help='policy class, capitalize, CNNMLP, ACT, Diffusion')
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], help='camera names')

    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')
    parser.add_argument('--use_robot_base', action='store_true', help='use robot base')

    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--lr', type=float, default=1e-5, help='lr')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--dilation', action='store_true',
                        help="replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', type=str, choices=('sine', 'learned'), default='sine',
                        help="type of positional embedding to use on top of the image features")
    parser.add_argument('--masks', action='store_true',
                        help="train segmentation head if the flag is provided")

    parser.add_argument('--kl_weight', type=int, default=10, help='KL Weight')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('--dim_feedforward', type=int, default=3200, help='dim feedforward')
    parser.add_argument('--temporal_agg', action='store_false', help='temporal agg')

    parser.add_argument('--state_dim', type=int, default=14, help='state dim')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='lr backbone')
    parser.add_argument('--backbone', type=str, default='resnet18', help='backbone')
    parser.add_argument('--loss_function', type=str, choices=['l1', 'l2', 'l1+l2'],
                        default='l1', help='loss function l1 l2 l1+l2')
    parser.add_argument('--enc_layers', type=int, default=4, help='enc layers')
    parser.add_argument('--dec_layers', type=int, default=7, help='dec layers')
    parser.add_argument('--nheads', type=int, default=8, help='nheads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="dropout applied in the transformer")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--chunk_size', type=int, default=30, help='chunk size')

    # for Diffusion
    parser.add_argument('--observation_horizon', type=int, default=1, help='observation horizon')
    parser.add_argument('--action_horizon', type=int, default=8, help='action horizon')
    parser.add_argument('--num_inference_timesteps', type=int, default=10,
                        help='num inference timesteps')
    parser.add_argument('--ema_power', type=int, default=0.75, help='ema power')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main():
    opt = parse_opt()
    config = get_model_config(opt)
    export_model(config)


if __name__ == '__main__':
    main()
