import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from utils.utils import load_data
from utils.utils import compute_dict_mean, set_seed, detach_dict
from utils.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy

import sys

sys.path.append("./")


def train(opt):
    set_seed(1)

    task_config = {
        'dataset_dir': os.path.expanduser(opt.datasets),
        'num_episodes': opt.num_episodes,
        'camera_names': opt.camera_names,
    }

    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    camera_names = task_config['camera_names']

    # fixed parameters
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
                         'camera_names': camera_names,
                         'use_depth_image': opt.use_depth_image,
                         'use_chassis': opt.use_chassis,
                         'kl_weight': opt.kl_weight,  # kl散度权重
                         'hidden_dim': opt.hidden_dim,  # 隐藏层维度
                         'dim_feedforward': opt.dim_feedforward,
                         'enc_layers': opt.enc_layers,
                         'dec_layers': opt.dec_layers,
                         'nheads': opt.nheads,
                         'dropout': opt.dropout,
                         'pre_norm': opt.pre_norm,
                         
                         'use_single_arm':opt.use_single_arm
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
                         'chunk_size': 1,  # 查询
                         'camera_names': camera_names,
                         'use_depth_image': opt.use_depth_image,
                         'use_chassis': opt.use_chassis,
                         'hidden_dim': opt.hidden_dim,
                         'use_single_arm':opt.use_single_arm
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
                         'camera_names': camera_names,
                         'use_depth_image': opt.use_depth_image,
                         'use_chassis': opt.use_chassis,
                         'observation_horizon': opt.observation_horizon,
                         'action_horizon': opt.action_horizon,
                         'num_inference_timesteps': opt.num_inference_timesteps,
                         'ema_power': opt.ema_power,
                         'hidden_dim': opt.hidden_dim,
                         'use_single_arm':opt.use_single_arm
                         }
    else:
        raise NotImplementedError

    config = {
        'num_epochs': opt.epochs,
        'ckpt_dir': opt.ckpt_dir,
        'ckpt_name': opt.ckpt_name,
        'policy_class': opt.policy_class,
        'policy_config': policy_config,
        'seed': opt.seed,
        'pretrain_ckpt': opt.pretrain_ckpt,
    }

    # data Preprocess
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, opt.arm_delay_time,
                                                           opt.use_depth_image, opt.use_chassis, camera_names,
                                                           opt.batch_size, opt.batch_size)

    # save dataset stats
    if not os.path.isdir(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir)
    stats_path = os.path.join(opt.ckpt_dir, opt.ckpt_stats_name)
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_process(train_dataloader, val_dataloader, config, stats)


def make_policy(policy_class, policy_config, pretrain_ckpt):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
        print(f"{pretrain_ckpt=}")
        if len(pretrain_ckpt) != 0 and pretrain_ckpt != '-1':
            
            state_dict = torch.load(pretrain_ckpt)
            print("trained on the pretrained ckpt:", pretrain_ckpt)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
                    continue
                # if policy_config['num_next_action'] == 0 and key in ["model.input_proj_next_action.weight",
                #                                                      "model.input_proj_next_action.bias"]:
                    continue
                new_state_dict[key] = value
            loading_status = policy.deserialize(new_state_dict)
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
        if len(pretrain_ckpt) != 0:
            loading_status = policy.deserialize(torch.load(pretrain_ckpt))
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
        if len(pretrain_ckpt) != 0:
            loading_status = policy.deserialize(torch.load(pretrain_ckpt))
            if not loading_status:
                print("ckpt path not exist")
    else:
        raise NotImplementedError

    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError

    return optimizer


def forward_pass(policy_config, data, policy):
    image_data, image_depth_data, qpos_data, action_data, action_is_pad = data
    (image_data, qpos_data, action_data, action_is_pad) = (image_data.cuda(), qpos_data.cuda(),
                                                           action_data.cuda(), action_is_pad.cuda())

    if policy_config['use_depth_image']:
        image_depth_data = image_depth_data.cuda()
    else:
        image_depth_data = None

    return policy(image_data, image_depth_data, qpos_data, action_data, action_is_pad)


def train_process(train_dataloader, val_dataloader, config, stats):
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']

    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    ckpt_name = config['ckpt_name']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    pretrain_ckpt = config['pretrain_ckpt']
    set_seed(seed)

    policy = make_policy(policy_class, policy_config, pretrain_ckpt)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    start_epoch = 0 
    
    # 加载已有的权重
    if pretrain_ckpt == '-1': # 继续最新的训练，同时保留 loss 历史数据
        for last_history_epoch in range(num_epochs-2,-1,-1):
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch{last_history_epoch + 1}_seed{seed}_pretrained_all_info.ckpt')
            
            if os.path.exists(ckpt_path): # Load the history trained weights of epoch
                print(f'Load the history trained weights of epoch={last_history_epoch+1}')
                checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
                policy.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
                optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
                start_epoch = checkpoint['epoch']  # 设置开始的epoch
                min_val_loss = checkpoint['min_val_loss']
                train_history = checkpoint['train_history']
                validation_history = checkpoint['validation_history']
                start_epoch = start_epoch + 1
                break 
    
        print(f"################### pretrained epoch={start_epoch} ###################")
        epoch = start_epoch
    
    best_ckpt_info = None
    for epoch in tqdm(range(start_epoch, num_epochs)):
        # print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict, result = forward_pass(policy_config, data, policy)
                # print("result:", post_process(result.cpu().detach().numpy())[0, :, 7:])
                epoch_dicts.append(forward_dict)

            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.serialize()))

                # save best checkpoint
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                torch.save(deepcopy(policy.serialize()), ckpt_path)
                # print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{epoch}')
        # print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        # for k, v in epoch_summary.items():
        #     summary_string += f'{k}: {v.item():.3f} '
        # print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict, result = forward_pass(policy_config, data, policy)
            # print("result:", post_process(result.cpu().detach().numpy())[0, :, 7:])
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)
        
    
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch{epoch + 1}_seed{seed}_pretrained_all_info.ckpt')
    checkpoint = {
        "net": policy.state_dict(),
        'optimizer':optimizer.state_dict(),
        "epoch": epoch,
        "min_val_loss": min_val_loss,
        "train_history": train_history,
        "validation_history": validation_history
    }
    torch.save(checkpoint, ckpt_path)
    
    
    # save best checkpoint
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)),
                 train_values, label='train')
        plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)),
                 val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets', type=str, default="./datasets", help='dataset dir')
    parser.add_argument('--ckpt_dir', type=str, default='./weights', help='ckpt dir')
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt', help='ckpt name')
    parser.add_argument('--pretrain_ckpt', type=str, default='', help='pretrain ckpt')
    parser.add_argument('--ckpt_stats_name', type=str, default='dataset_stats.pkl',
                        help='ckpt stats name')

    parser.add_argument('--num_episodes', type=int, default='50', help='episodes number')
    parser.add_argument('--policy_class', type=str, choices=['CNNMLP', 'ACT', 'Diffusion'],
                        default='ACT', help='policy class, capitalize, CNNMLP, ACT, Diffusion')
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                        default=['cam_left_wrist'], help='camera names')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--epochs', type=int, default=3000, help='epochs number')

    parser.add_argument('--lr', type=float, default=4e-5, help='lr')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--dilation', action='store_true',
                        help="replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', type=str, choices=('sine', 'learned'),
                        default='sine', help="type of positional embedding to use on top of the image features")
    parser.add_argument('--masks', action='store_true',
                        help="train segmentation head if the flag is provided")

    parser.add_argument('--state_dim', type=int, default=14, help='state dim')
    parser.add_argument('--lr_backbone', type=float, default=4e-5, help='lr backbone')
    parser.add_argument('--backbone', type=str, default='resnet18', help='backbone')
    parser.add_argument('--loss_function', type=str, choices=['l1', 'l2', 'l1+l2'],
                        default='l1', help='loss function l1 l2 l1+l2')
    parser.add_argument('--enc_layers', type=int, default=4, help='enc_layers')
    parser.add_argument('--dec_layers', type=int, default=7, help='dec_layers')
    parser.add_argument('--nheads', type=int, default=8, help='nheads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="dropout applied in the transformer")
    parser.add_argument('--pre_norm', action='store_true')

    # for ACT
    parser.add_argument('--kl_weight', type=int, default=10, help='KL Weight')
    parser.add_argument('--chunk_size', type=int, default=30, help='chunk size')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('--dim_feedforward', type=int, default=3200, help='dim feedforward')
    parser.add_argument('--temporal_agg', type=bool, default=True, help='temporal agg')

    # for Diffusion
    parser.add_argument('--observation_horizon', type=int, default=1, help='observation horizon')
    parser.add_argument('--action_horizon', type=int, default=8, help='action horizon')
    parser.add_argument('--num_inference_timesteps', type=int, default=10,
                        help='num inference timesteps')
    parser.add_argument('--ema_power', type=int, default=0.75, help='ema power')

    parser.add_argument('--arm_delay_time', type=int, default=0, help='arm delay time')

    parser.add_argument('--use_chassis', action='store_true', help='use robot base')

    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')
    parser.add_argument('--use_single_arm', action='store_true', help='if use single arm') # 是否使用单臂遥操作（一个主臂一个从臂）

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main():
    opt = parse_opt()
    train(opt)


if __name__ == '__main__':
    main()
