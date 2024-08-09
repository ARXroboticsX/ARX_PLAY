# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import argparse
import collections
import math
import os
import pickle
import sys
import threading
import time
import yaml
from collections import deque

import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from einops import rearrange
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header

from utils.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from utils.utils import set_seed  # helper functions

from msg._JointControl import JointControl
from msg._JointInformation import JointInformation
from msg._PosCmd import PosCmd
# import cv2 as cv

sys.path.append("./")

# 为多线程的全局变量，
inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def get_model_config(opt):
    # 设置随机种子，你可以确保在相同的初始条件下，每次运行代码时生成的随机数序列是相同的。
    set_seed(1)

    # 如果是ACT策略
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
                         'camera_names': opt.camera_names,
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
                         'camera_names': opt.camera_names,
                         'use_depth_image': opt.use_depth_image,
                         'use_chassis': opt.use_chassis,
                         
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
                         'camera_names': opt.camera_names,
                         'use_depth_image': opt.use_depth_image,
                         'use_chassis': opt.use_chassis,
                         'observation_horizon': opt.observation_horizon,
                         'action_horizon': opt.action_horizon,
                         'num_inference_timesteps': opt.num_inference_timesteps,
                         'ema_power': opt.ema_power,
                         
                         'use_single_arm':opt.use_single_arm
                         }
    else:
        raise NotImplementedError

    config = {
        'ckpt_dir': opt.ckpt_dir,
        'ckpt_name': opt.ckpt_name,
        'ckpt_stats_name': opt.ckpt_stats_name,
        'episode_len': opt.max_publish_step,
        'state_dim': 7 if opt.use_single_arm else 14,
        'policy_class': opt.policy_class,
        'policy_config': policy_config,
        'temporal_agg': opt.temporal_agg,
        'camera_names': opt.camera_names,
    }
    if opt.use_chassis:
        config['state_dim'] = config['state_dim'] + 2
    
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


def inference_process(opt, config, ros_operator, policy, stats, t, use_single_arm):
    global inference_lock
    global inference_actions
    global inference_timestep

    print_flag = True
    pre_pos_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    rate = rospy.Rate(opt.publish_rate)
    
    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_left, img_right, img_head, img_left_depth, img_right_depth, img_head_depth, 
         follow_arm_left, follow_arm_right, follow_arm_left_eef, follow_arm_right_eef,
         robot_base) = result
        
        obs = collections.OrderedDict()

        camera_map = {
            'cam_left_wrist': img_left,
            'cam_right_wrist': img_right,
            'cam_high': img_head,
        }

        image_dict = dict()
        for camera_name, image in camera_map.items():
            if camera_name in config['camera_names']:
                image_dict[camera_name] = image

        obs['images'] = image_dict

        if opt.use_depth_image:
            camera_depth_map = {
                'cam_left_wrist': img_left_depth,
                'cam_right_wrist': img_right_depth,
                'cam_high': img_head_depth,
            }

            image_depth_dict = dict()
            for camera_name, image in camera_depth_map.items():
                if camera_name in config['camera_names']:
                    image_depth_dict[camera_name] = image

            obs['images_depth'] = image_depth_dict
        follow_arm_left_eef_array = [follow_arm_left_eef.x, follow_arm_left_eef.y, follow_arm_left_eef.z,
                                          follow_arm_left_eef.roll, follow_arm_left_eef.pitch, follow_arm_left_eef.yaw, follow_arm_left_eef.gripper]
        
        if not opt.use_single_arm:
            follow_arm_right_eef_array = [follow_arm_right_eef.x, follow_arm_right_eef.y, follow_arm_right_eef.z,
                                            follow_arm_right_eef.roll, follow_arm_right_eef.pitch, follow_arm_right_eef.yaw, follow_arm_right_eef.gripper]
            obs['eef'] = np.concatenate((np.array(follow_arm_left_eef_array),np.array(follow_arm_right_eef_array)), axis=0)
            obs['qpos'] = np.concatenate((np.array(follow_arm_left.joint_pos),np.array(follow_arm_right.joint_pos)), axis=0)
            obs['qvel'] = np.concatenate((np.array(follow_arm_left.joint_vel),np.array(follow_arm_right.joint_vel)), axis=0)
            obs['effort'] = np.concatenate((np.array(follow_arm_left.joint_cur),np.array(follow_arm_right.joint_cur)), axis=0)
        else:
            obs['qpos'] = np.array(follow_arm_left.joint_pos)
            obs['qvel'] = np.array(follow_arm_left.joint_vel)
            obs['effort'] = np.array(follow_arm_left.joint_cur)
            obs['eef'] = follow_arm_left_eef_array    

        if opt.use_chassis:
            obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]

        # 归一化处理qpos 并转到cuda
        qpos = pre_pos_process(obs['qpos'])
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

        # 当前图像curr_image获取图像
        curr_image = get_image(obs, config['camera_names'])
        curr_depth_image = None

        if opt.use_depth_image:
            curr_depth_image = get_depth_image(obs, config['camera_names'])

        # 模型推理
        # start_time = time.time()
        all_actions = policy(curr_image, curr_depth_image, qpos)########################
        # end_time = time.time()
        # print(inference_timestep ,"model cost time: ", end_time - start_time)

        inference_lock.acquire()
        inference_actions = all_actions.cpu().detach().numpy() # global var

        inference_timestep = t
        inference_lock.release()
        break

def model_inference(opt, config, ros_operator):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread
    set_seed(1000)
    
    use_single_arm = config['policy_config']['use_single_arm']
    # 1 创建模型数据  继承nn.Module
    policy = make_policy(config['policy_class'], config['policy_config'])

    # 2 加载模型权重
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

    # 3 模型设置为cuda模式和验证模式
    policy.cuda()
    policy.eval()

    # 4 加载统计值
    stats_path = os.path.join(config['ckpt_dir'], config['ckpt_stats_name'])
    # 统计的数据  # 加载action_mean, action_std, qpos_mean, qpos_std 14维
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 数据预处理和后处理函数定义
    post_process = lambda a: a * stats['action_std'] + stats['action_mean'] # 正确对应的

    max_publish_step = config['episode_len']
    chunk_size = config['policy_config']['chunk_size']

    # 发布基础的姿态
    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375,
             -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656,
              -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
    left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375,
             -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
    right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375,
              -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]

    ros_operator.follow_arm_publish_continuous(left0, right0)
    input("Enter any key to continue :")
    ros_operator.follow_arm_publish_continuous(left1, right1)
    action = None
    
    # 初始化显示一下
    
    # 推理
    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            # 每个回合的步数
            t = 0
            max_t = 0
            rate = rospy.Rate(opt.publish_rate)

            if config['temporal_agg']:
                print(f"{config['state_dim']=}")
                
                action_dim = config['state_dim']
                all_time_actions = np.zeros([max_publish_step, max_publish_step + chunk_size, action_dim]) # (10000 , 10100, 14)

            while t < max_publish_step and not rospy.is_shutdown():
                # query policy
                if config['policy_class'] == "ACT":
                    if t >= max_t:
                        pre_action = action
                        inference_thread = threading.Thread(target=inference_process,
                                                            args=(opt, config, ros_operator,
                                                                  policy, stats, t, use_single_arm))
                        inference_thread.start()
                        inference_thread.join()
                        inference_lock.acquire()
                        if inference_actions is not None:
                            inference_thread = None
                            all_actions = inference_actions
                            inference_actions = None
                            max_t = t + opt.pos_lookahead_step
                            if config['temporal_agg']:
                                all_time_actions[[t], t:t + chunk_size] = all_actions
                        inference_lock.release()
                    if config['temporal_agg']:
                        actions_for_curr_step = all_time_actions[:, t] # (10000,1,14) => (10000, 14)
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        if opt.pos_lookahead_step != 0:
                            raw_action = all_actions[:, t % opt.pos_lookahead_step]
                        else:
                            raw_action = all_actions[:, t % chunk_size]
                else:
                    raise NotImplementedError
                action = post_process(raw_action[0])
                
                print(f"{t=}, \t {action=}")
                left_action = action[:7]  # 取7维度
                if not use_single_arm:
                    right_action = action[7:14]
                else:
                    right_action = []
                ros_operator.follow_arm_publish(left_action, right_action)  # follow_arm_publish_continuous_thread

                if opt.use_chassis:
                    vel_action = action[14:16]
                    ros_operator.robot_base_publish(vel_action)

                t += 1
                end_time = time.time()

                rate.sleep()


class RosOperator:
    def __init__(self, opt, config):
        self.robot_base_deque = None
        self.follow_arm_right_deque = None
        self.follow_arm_left_deque = None
        
        # eef
        self.follow_arm_right_eef_deque = None
        self.follow_arm_left_eef_deque = None
        
        self.img_head_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_head_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.follow_arm_left_publisher = None
        self.follow_arm_right_publisher = None
        self.robot_base_publisher = None
        self.follow_arm_publish_thread = None
        self.follow_arm_publish_lock = None
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.opt = opt
        self.config = config
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_head_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_head_depth_deque = deque()
        self.follow_arm_left_deque = deque()
        self.follow_arm_right_deque = deque()
        
        # eef
        self.follow_arm_right_eef_deque = deque()
        self.follow_arm_left_eef_deque = deque()
        
        self.robot_base_deque = deque()
        self.follow_arm_publish_lock = threading.Lock()
        self.follow_arm_publish_lock.acquire()

    def follow_arm_publish(self, left, right):
        joint_state_msg = JointControl()
        joint_state_msg.joint_pos = left
        self.follow_arm_left_publisher.publish(joint_state_msg)
        if len(right) != 0:
            joint_state_msg.joint_pos = right
            self.follow_arm_right_publisher.publish(joint_state_msg)
        

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def follow_arm_publish_continuous(self, left_target, right_target): # left个right都是固定的位置
        rate = rospy.Rate(self.opt.publish_rate)
        left_arm = None
        right_arm = None
        
        # 判断是否还有步数没走完
        while True and not rospy.is_shutdown():
            if len(self.follow_arm_left_deque) != 0:
                left_arm = list(self.follow_arm_left_deque[-1].joint_pos)
            if len(self.follow_arm_right_deque) != 0:
                right_arm = list(self.follow_arm_right_deque[-1].joint_pos)
                
            if left_arm is None and right_arm is None:
                rate.sleep()
                continue
            
            else:
                break
        
        # 是否完成位移的标志位
        left_symbol = [1 if left_target[i] - left_arm[i] > 0 else -1 for i in range(len(left_target))]
        if right_arm:
            right_symbol = [1 if right_target[i] - right_arm[i] > 0 else -1 for i in range(len(right_target))]
        flag = True
        step = 0

        while flag and not rospy.is_shutdown():
            right_done = 0
            left_done = 0

            if self.follow_arm_publish_lock.acquire(False):
                return

            left_diff = [abs(left_target[i] - left_arm[i]) for i in range(len(left_target))]
            for i in range(len(left_target)):
                if left_diff[i] < self.opt.arm_steps_length[i]:
                    left_arm[i] = left_target[i]
                    left_done = left_done + 1
                else:
                    left_arm[i] += left_symbol[i] * self.opt.arm_steps_length[i]

            if right_arm:
                right_diff = [abs(right_target[i] - right_arm[i]) for i in range(len(right_target))]
                for i in range(len(right_target)):
                    if right_diff[i] < self.opt.arm_steps_length[i]:
                        right_arm[i] = right_target[i]
                        right_done = right_done + 1
                    else:
                        right_arm[i] += right_symbol[i] * self.opt.arm_steps_length[i]

            if right_arm:
                if left_done > len(left_target) - 1 and right_done > len(right_target) - 1:
                    print('left_done and right_done')
                    break
            elif left_done > len(left_target) - 1:
                break

            joint_state_msg = JointControl()
            joint_state_msg.joint_pos = left_arm
            self.follow_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.joint_pos = right_arm
            self.follow_arm_right_publisher.publish(joint_state_msg)

            step += 1
            print("follow_arm_publish_continuous:", step)
            rate.sleep()

    def follow_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.follow_arm_left_deque) != 0:
                left_arm = list(self.follow_arm_left_deque[-1].joint_pos)
            if len(self.follow_arm_right_deque) != 0:
                right_arm = list(self.follow_arm_right_deque[-1].joint_pos)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3',
                                    'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.joint_pos = traj_left
            self.follow_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.joint_pos = traj_right
            self.follow_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def follow_arm_publish_continuous_thread(self, left, right):
        if self.follow_arm_publish_thread is not None:
            self.follow_arm_publish_lock.release()
            self.follow_arm_publish_thread.join()
            self.follow_arm_publish_lock.acquire(False)
            self.follow_arm_publish_thread = None
        self.follow_arm_publish_thread = threading.Thread(target=self.follow_arm_publish_continuous,
                                                          opt=(left, right))
        self.follow_arm_publish_thread.start()

    def get_frame(self):
        img_left = []
        img_right = []
        img_head = []
        img_left_depth = []
        img_right_depth = []
        img_head_depth = []
        robot_base = None
        follow_arm_right = None
        follow_arm_right_eef = None

        if 'cam_left_wrist' in self.opt.camera_names:
            img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.pop(), 'passthrough')
        if 'cam_right_wrist' in self.opt.camera_names:   
            img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.pop(), 'passthrough')
        if 'cam_head_wrist' in self.opt.camera_names:
            img_head = self.bridge.imgmsg_to_cv2(self.img_head_deque.pop(), 'passthrough')
        
        if self.opt.use_depth_image: # not recommend
            if 'cam_left_wrist' in self.opt.camera_names:
                img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.pop(), 'passthrough')
            if 'cam_right_wrist' in self.opt.camera_names:
                img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.pop(), 'passthrough')
            if 'cam_head_wrist' in self.opt.camera_names:
                img_head_depth = self.bridge.imgmsg_to_cv2(self.img_head_depth_deque.pop(), 'passthrough')
                
        follow_arm_left = self.follow_arm_left_deque.pop()
        follow_arm_left_eef = self.follow_arm_left_eef_deque.pop() # 模型输出
        if not self.opt.use_single_arm:
            follow_arm_right = self.follow_arm_right_deque.pop()
            follow_arm_right_eef = self.follow_arm_right_eef_deque.pop()
        
        if self.opt.use_chassis:
            robot_base = self.robot_base_deque.pop()
            
        return (img_left, img_right, img_head, img_left_depth, img_right_depth, img_head_depth, 
                follow_arm_left, follow_arm_right, follow_arm_left_eef, follow_arm_right_eef,
                robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_head_callback(self, msg):
        if len(self.img_head_deque) >= 2000:
            self.img_head_deque.popleft()
        self.img_head_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_head_depth_callback(self, msg):
        if len(self.img_head_depth_deque) >= 2000:
            self.img_head_depth_deque.popleft()
        self.img_head_depth_deque.append(msg)

    def follow_arm_left_callback(self, msg):
        if len(self.follow_arm_left_deque) >= 2000:
            self.follow_arm_left_deque.popleft()
        self.follow_arm_left_deque.append(msg)

    def follow_arm_right_callback(self, msg):
        if len(self.follow_arm_right_deque) >= 2000:
            self.follow_arm_right_deque.popleft()
        self.follow_arm_right_deque.append(msg)
    
    # eef
    def master_arm_left_eef_callback(self, msg):
        if len(self.master_arm_left_eef_deque) >= 2000:
            self.master_arm_left_eef_deque.popleft()
        self.master_arm_left_eef_deque.append(msg)

    def master_arm_right_eef_callback(self, msg):
        if len(self.master_arm_right_eef_deque) >= 2000:
            self.master_arm_right_eef_deque.popleft()
        self.master_arm_right_eef_deque.append(msg)

    def follow_arm_left_eef_callback(self, msg):
        if len(self.follow_arm_left_eef_deque) >= 2000:
            self.follow_arm_left_eef_deque.popleft()
        self.follow_arm_left_eef_deque.append(msg)
        # print(len(self.follow_arm_left_eef_deque))

    def follow_arm_right_eef_callback(self, msg):
        if len(self.follow_arm_right_eef_deque) >= 2000:
            self.follow_arm_right_eef_deque.popleft()
        self.follow_arm_right_eef_deque.append(msg)
    
    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)

        rospy.Subscriber(self.config['camera_config']['original_image']['img_left_topic'],
                         Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['camera_config']['original_image']['img_right_topic'],
                         Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['camera_config']['original_image']['img_head_topic'],
                         Image, self.img_head_callback, queue_size=1000, tcp_nodelay=True)

        if self.opt.use_depth_image:
            rospy.Subscriber(self.config['camera_config']['original_image']['img_left_depth_topic'],
                             Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.config['camera_config']['original_image']['img_right_depth_topic'],
                             Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.config['camera_config']['original_image']['img_head_depth_topic'],
                             Image, self.img_head_depth_callback, queue_size=1000, tcp_nodelay=True)

        rospy.Subscriber(self.config['arm_config']['follow_arm_left_topic'],
                         JointInformation, self.follow_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['arm_config']['follow_arm_right_topic'],
                         JointInformation, self.follow_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['base_config']['robot_base_topic'],
                         Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        
        # more information
        # eef
        rospy.Subscriber(self.config['arm_config']['follow_arm_left_eef_topic'],
                         PosCmd, self.follow_arm_left_eef_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['arm_config']['follow_arm_right_eef_topic'],
                         PosCmd, self.follow_arm_right_eef_callback, queue_size=1000, tcp_nodelay=True)

        self.follow_arm_left_publisher = rospy.Publisher(self.config['arm_config']['follow_arm_left_cmd_topic'],
                                                         JointControl, queue_size=10)
        self.follow_arm_right_publisher = rospy.Publisher(self.config['arm_config']['follow_arm_right_cmd_topic'],
                                                          JointControl, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.config['base_config']['robot_base_cmd_topic'],
                                                    Twist, queue_size=10)



def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_dir', type=str, help='ckpt dir', required=True)
    parser.add_argument('--max_publish_step', type=int, default=10000, help='max publish step')
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt', help='ckpt name')
    parser.add_argument('--ckpt_stats_name', type=str, default='dataset_stats.pkl',
                        help='ckpt stats name')
    parser.add_argument('--policy_class', type=str, choices=['CNNMLP', 'ACT', 'Diffusion'],
                        default='ACT', help='policy class, capitalize, CNNMLP, ACT, Diffusion')
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                        default=['cam_left_wrist'], help='camera names')

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

    parser.add_argument('--data', type=str, default="./data/config.yaml", help='config file')

    parser.add_argument('--use_chassis', action='store_true', help='use robot base')

    parser.add_argument('--publish_rate', type=int, default=90, help='publish rate')
    parser.add_argument('--pos_lookahead_step', type=int, default=0, help='pos lookahead step')
    parser.add_argument('--chunk_size', type=int, default=30, help='chunk size')
    parser.add_argument('--arm_steps_length', nargs='+', type=float,
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], help='arm_steps_length')

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='use actions interpolation')
    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')

    # for Diffusion
    parser.add_argument('--observation_horizon', type=int, default=1, help='observation horizon')
    parser.add_argument('--action_horizon', type=int, default=8, help='action horizon')
    parser.add_argument('--num_inference_timesteps', type=int, default=10,
                        help='num inference timesteps')
    parser.add_argument('--ema_power', type=int, default=0.75, help='ema power')
    parser.add_argument('--use_single_arm', action='store_true', help='if use single arm') # 是否使用单臂遥操作（一个主臂一个从臂）

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main():
    np.set_printoptions(linewidth=200)

    opt = parse_opt() # 参数设置
    data = load_yaml(opt.data) 
    ros_operator = RosOperator(opt, data) # 订阅消息和发布器初始化
    config = get_model_config(opt) # 获取模型参数
    model_inference(opt, config, ros_operator) # 模型推理


if __name__ == '__main__':
    main()
