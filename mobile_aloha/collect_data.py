# -- coding: UTF-8
import os
import time
import numpy as np
import h5py
import argparse
import dm_env
import collections
from collections import deque
import rospy
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import sys
import cv2
import yaml

from msg._JointControl import JointControl
from msg._JointInformation import JointInformation

from msg._PosCmd import PosCmd

# 读取相关配置文件
def load_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def compress_images(image_list, encode_param):
    compressed_list = []
    compressed_len = []

    for image in image_list:
        result, encoded_image = cv2.imencode('.jpg', image, encode_param)
        compressed_list.append(encoded_image)
        compressed_len.append(len(encoded_image))

    return compressed_list, compressed_len


def pad_images(compressed_image_list, padded_size):
    padded_compressed_image_list = []

    for compressed_image in compressed_image_list:
        padded_compressed_image = np.zeros(padded_size, dtype='uint8')
        image_len = len(compressed_image)
        padded_compressed_image[:image_len] = compressed_image
        padded_compressed_image_list.append(padded_compressed_image)

    return padded_compressed_image_list


# 保存数据函数
def save_data(opt, timesteps, actions, actions_eef, dataset_path, use_single_arm):
    # 数据字典
    data_size = len(actions)
    data_dict = {
        # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/observations/eef': [],
        '/action': [],
        '/action_eef': [],
        '/base_action': [],
    }

    # 相机字典  观察的图像
    for cam_name in opt.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if opt.use_depth_image:
            data_dict[f'/observations/depths/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    # 动作长度 遍历动作
    while actions:
        # 循环弹出一个队列
        action = actions.pop(0)  # 动作  当前动作
        action_eef = actions_eef.pop(0)
        ts = timesteps.pop(0)  # 奖励  前一帧

        # 往字典里面添值
        # Timestep返回的qpos，qvel,effort
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/observations/eef'].append(ts.observation['eef'])
        
        # 实际发的action
        data_dict['/action'].append(action)
        data_dict['/action_eef'].append(action_eef)
        data_dict['/base_action'].append(ts.observation['base_vel'])

        # 相机数据
        for cam_name in opt.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(
                ts.observation['images'][cam_name])
            if opt.use_depth_image:
                data_dict[f'/observations/depths/{cam_name}'].append(
                    ts.observation['depths'][cam_name])

    # 压缩图像
    if opt.is_compress:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # 压缩质量
        compressed_len = []

        for cam_name in opt.camera_names:
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

        for cam_name in opt.camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)

            # 更新压缩后的图像列表
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list

        if opt.use_depth_image:
            compressed_len_depth = []

            for cam_name in opt.camera_names:
                depth_list = data_dict[f'/observations/depths/{cam_name}']
                compressed_list_depth = []
                compressed_len_depth.append([])  # 压缩的长度

                for depth in depth_list:
                    result, encoded_depth = cv2.imencode('.jpg', depth, encode_param)

                    compressed_list_depth.append(encoded_depth)
                    compressed_len_depth[-1].append(len(encoded_depth))

                # 更新图像
                data_dict[f'/observations/depths/{cam_name}'] = compressed_list_depth

            compressed_len_depth = np.array(compressed_len_depth)
            padded_size_depth = compressed_len_depth.max()

            for cam_name in opt.camera_names:
                compressed_depth_list = data_dict[f'/observations/depths/{cam_name}']
                padded_compressed_depth_list = []
                for compressed_depth in compressed_depth_list:
                    padded_compressed_depth = np.zeros(padded_size_depth, dtype='uint8')
                    depth_len = len(compressed_depth)
                    padded_compressed_depth[:depth_len] = compressed_depth
                    padded_compressed_depth_list.append(padded_compressed_depth)
                data_dict[f'/observations/depths/{cam_name}'] = padded_compressed_depth_list

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        # 文本的属性：
        # 1 是否仿真
        # 2 图像是否压缩

        root.attrs['sim'] = False
        root.attrs['compress'] = False
        if opt.is_compress:
            root.attrs['compress'] = True

        # 创建一个新的组observations，观测状态组
        # 图像组
        obs = root.create_group('observations')
        image = obs.create_group('images')
        depth = obs.create_group('depths')

        for cam_name in opt.camera_names:
            if opt.is_compress:
                image_shape = (opt.max_timesteps, padded_size)
                image_chunks = (1, padded_size)

                if opt.use_depth_image:
                    depth_shape = (opt.max_timesteps, padded_size_depth)
                    depth_chunks = (1, padded_size_depth)
            else:
                image_shape = (opt.max_timesteps, 480, 640, 3)
                image_chunks = (1, 480, 640, 3)

                if opt.use_depth_image:
                    depth_shape = (opt.max_timesteps, 480, 640)
                    depth_chunks = (1, 480, 640)

            _ = image.create_dataset(cam_name, image_shape, 'uint8', chunks=image_chunks)
            if opt.use_depth_image:
                _ = depth.create_dataset(cam_name, depth_shape, 'uint8', chunks=depth_chunks)
        
        states_dim = 14
        if use_single_arm:
            states_dim = 7
            
        _ = obs.create_dataset('qpos', (data_size, states_dim))
        _ = obs.create_dataset('eef', (data_size, states_dim))
        _ = obs.create_dataset('qvel', (data_size, states_dim))
        _ = obs.create_dataset('effort', (data_size, states_dim))
        _ = root.create_dataset('action', (data_size, states_dim))
        _ = root.create_dataset('action_eef', (data_size, states_dim))
        _ = root.create_dataset('base_action', (data_size, 2))

        # data_dict写入h5py.File
        for name, array in data_dict.items():  # 名字+值
            root[name][...] = array

    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n'%dataset_path)


class RosOperator:
    def __init__(self, opt, config):
        self.robot_base_deque = None
        self.follow_arm_right_deque = None
        self.follow_arm_left_deque = None
        self.master_arm_right_deque = None
        self.master_arm_left_deque = None
        
        # eef
        self.follow_arm_right_eef_deque = None
        self.follow_arm_left_eef_deque = None
        self.master_arm_right_eef_deque = None
        self.master_arm_left_eef_deque = None
        
        self.img_head_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_head_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
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
        self.master_arm_left_deque = deque()
        self.master_arm_right_deque = deque()
        self.follow_arm_left_deque = deque()
        self.follow_arm_right_deque = deque()
        
        # eef
        self.follow_arm_right_eef_deque = deque()
        self.follow_arm_left_eef_deque = deque()
        self.master_arm_right_eef_deque = deque()
        self.master_arm_left_eef_deque = deque()
        
        self.robot_base_deque = deque()

    def get_frame(self): # if double arm need the head or high or head camera    
        img_left = []
        img_right = []
        img_head = []
        img_left_depth = []
        img_right_depth = []
        img_head_depth = []
        robot_base = None
        
        if self.opt.is_compress:
            if 'cam_left_wrist' in self.opt.camera_names:
                img_left = self.bridge.compressed_imgmsg_to_cv2(self.img_left_deque.pop(), 'passthrough')
                
            if 'cam_right_wrist' in self.opt.camera_names:
                img_right = self.bridge.compressed_imgmsg_to_cv2(self.img_right_deque.pop(), 'passthrough')
                
            if 'cam_head' in self.opt.camera_names:
                img_head = self.bridge.compressed_imgmsg_to_cv2(self.img_head_deque.pop(), 'passthrough') 
            
        else: # not recommend
            if 'cam_left_wrist' in self.opt.camera_names:
                img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.pop(), 'passthrough')

            if 'cam_right_wrist' in self.opt.camera_names:
                img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.pop(), 'passthrough')

            if 'cam_head' in self.opt.camera_names:
                img_head = self.bridge.imgmsg_to_cv2(self.img_head_deque.pop(), 'passthrough')
        
        if self.opt.use_depth_image: # not recommend
            if self.opt.is_compress:
                if 'cam_left_wrist' in self.opt.camera_names:
                    img_left_depth = self.bridge.compressed_imgmsg_to_cv2(self.img_left_depth_deque.pop(),
                                                                          'passthrough')

                if 'cam_right_wrist' in self.opt.camera_names:
                    img_right_depth = self.bridge.compressed_imgmsg_to_cv2(self.img_right_depth_deque.pop(),
                                                                           'passthrough')

                if 'cam_head' in self.opt.camera_names:
                    img_head_depth = self.bridge.compressed_imgmsg_to_cv2(self.img_head_depth_deque.pop(),
                                                                          'passthrough')
            else:
                if 'cam_left_wrist' in self.opt.camera_names:
                    img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.pop(),
                                                                          'passthrough')

                if 'cam_right_wrist' in self.opt.camera_names:
                    img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.pop(),
                                                                           'passthrough')

                if 'cam_head' in self.opt.camera_names:
                    img_head_depth = self.bridge.imgmsg_to_cv2(self.img_head_depth_deque.pop(),
                                                                          'passthrough')

        master_arm_left = self.master_arm_left_deque.pop()
        master_arm_left_eef = self.master_arm_left_eef_deque.pop() 
        
        follow_arm_left = self.follow_arm_left_deque.pop()
        follow_arm_left_eef = self.follow_arm_left_eef_deque.pop() 
        
        master_arm_right = None
        master_arm_right_eef = None
        follow_arm_right = None
        follow_arm_right_eef = None
        if not self.opt.use_single_arm:
            master_arm_right_eef = self.master_arm_right_eef_deque.pop()
            master_arm_right = self.master_arm_right_deque.pop()
            follow_arm_right_eef = self.follow_arm_right_eef_deque.pop()
            follow_arm_right = self.follow_arm_right_deque.pop()

        if self.opt.use_chassis:
            robot_base = self.robot_base_deque.pop()
            
        return (img_left, img_right, img_head, img_left_depth, img_right_depth, img_head_depth, 
                follow_arm_left, follow_arm_right, follow_arm_left_eef, follow_arm_right_eef,
                master_arm_left, master_arm_right, master_arm_left_eef, master_arm_right_eef,
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

    def master_arm_left_callback(self, msg):
        if len(self.master_arm_left_deque) >= 2000:
            self.master_arm_left_deque.popleft()
        self.master_arm_left_deque.append(msg)

    def master_arm_right_callback(self, msg):
        if len(self.master_arm_right_deque) >= 2000:
            self.master_arm_right_deque.popleft()
        self.master_arm_right_deque.append(msg)

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

    def follow_arm_right_eef_callback(self, msg):
        if len(self.follow_arm_right_eef_deque) >= 2000:
            self.follow_arm_right_eef_deque.popleft()
        self.follow_arm_right_eef_deque.append(msg)
        

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node('record_episodes', anonymous=True)

        image_type = 'compress_image' if self.opt.is_compress else 'original_image'
        callback_type = CompressedImage if self.opt.is_compress else Image

        rospy.Subscriber(self.config['camera_config'][image_type]['img_left_topic'],
                         callback_type, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['camera_config'][image_type]['img_right_topic'],
                         callback_type, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['camera_config'][image_type]['img_head_topic'],
                         callback_type, self.img_head_callback, queue_size=1000, tcp_nodelay=True)

        if self.opt.use_depth_image:
            rospy.Subscriber(self.config['camera_config'][image_type]['img_left_depth_topic'],
                             callback_type, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.config['camera_config'][image_type]['img_right_depth_topic'],
                             callback_type, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.config['camera_config'][image_type]['img_head_depth_topic'],
                             callback_type, self.img_head_depth_callback, queue_size=1000, tcp_nodelay=True)

        rospy.Subscriber(self.config['arm_config']['master_arm_left_topic'],
                         JointControl, self.master_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['arm_config']['master_arm_right_topic'],
                         JointControl, self.master_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['arm_config']['follow_arm_left_topic'],
                         JointInformation, self.follow_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['arm_config']['follow_arm_right_topic'],
                         JointInformation, self.follow_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        
        # more information
        # eef
        rospy.Subscriber(self.config['arm_config']['master_arm_left_eef_topic'],
                         PosCmd, self.master_arm_left_eef_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['arm_config']['master_arm_right_eef_topic'],
                         PosCmd, self.master_arm_right_eef_callback, queue_size=1000, tcp_nodelay=True)
        
        rospy.Subscriber(self.config['arm_config']['follow_arm_left_eef_topic'],
                         PosCmd, self.follow_arm_left_eef_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.config['arm_config']['follow_arm_right_eef_topic'],
                         PosCmd, self.follow_arm_right_eef_callback, queue_size=1000, tcp_nodelay=True)
        

        rospy.Subscriber(self.config['base_config']['robot_base_topic'],
                         Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)

    def process(self):
        timesteps = []
        actions = []
        actions_eef =[]
        # 图像数据
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        image_dict = dict()
        for cam_name in self.opt.camera_names:
            image_dict[cam_name] = image
        count = 0

        rate = rospy.Rate(self.opt.frame_rate)
        print_flag = True
        
        for i in range(3, -1, -1):
            print(f"\rwaiting {i} to start recording",end='')
            rospy.sleep(0.2)
        
        global exit_flag
        while (count < self.opt.max_timesteps + 1) and not rospy.is_shutdown():
            # 2 收集数据
            result = self.get_frame()
            if not result:
                if print_flag:
                    print("syn fail")
                    print_flag = False
                rate.sleep()
                continue
            elif result == 0: # wait for camera frame
                rate.sleep()
                continue

            print_flag = True
            count += 1
            (img_left, img_right, img_head, img_left_depth, img_right_depth, img_head_depth, 
             follow_arm_left, follow_arm_right, follow_arm_left_eef, follow_arm_right_eef,
             master_arm_left, master_arm_right, master_arm_left_eef, master_arm_right_eef,
             robot_base) = result

            # 2.1 图像信息
            camera_map = {
                'cam_left_wrist': img_left,
                'cam_right_wrist': img_right,
                'cam_head': img_head,
            }

            image_dict = dict()
            for camera_name, image in camera_map.items():
                if camera_name in self.opt.camera_names:
                    image_dict[camera_name] = image

            # 2.2 从臂的信息从臂的状态 机械臂示教模式时 会自动订阅
            obs = collections.OrderedDict()  # 有序的字典
            obs['images'] = image_dict
            if self.opt.use_depth_image:
                image_dict_depth = dict()
                image_dict_depth[self.opt.camera_names[0]] = img_left_depth
                image_dict_depth[self.opt.camera_names[1]] = img_right_depth
                image_dict_depth[self.opt.camera_names[2]] = img_head_depth
                obs['depths'] = image_dict_depth
            
            follow_arm_left_eef_array = [follow_arm_left_eef.x, follow_arm_left_eef.y, follow_arm_left_eef.z,
                                            follow_arm_left_eef.roll, follow_arm_left_eef.pitch, follow_arm_left_eef.yaw, follow_arm_left_eef.gripper]
                
            if not self.opt.use_single_arm:
                follow_arm_right_eef_array = [follow_arm_right_eef.x, follow_arm_right_eef.y, follow_arm_right_eef.z,
                                            follow_arm_right_eef.roll, follow_arm_right_eef.pitch, follow_arm_right_eef.yaw, follow_arm_right_eef.gripper]
                obs['qpos'] = np.concatenate((np.array(follow_arm_left.joint_pos), np.array(follow_arm_right.joint_pos)), axis=0)
                obs['qvel'] = np.concatenate((np.array(follow_arm_left.joint_vel), np.array(follow_arm_right.joint_vel)), axis=0)
                obs['effort'] = np.concatenate((np.array(follow_arm_left.joint_cur), np.array(follow_arm_right.joint_cur)), axis=0)
                obs['eef'] = np.concatenate((follow_arm_left_eef_array, follow_arm_right_eef_array), axis=0)
            else:
                obs['qpos'] = np.array(follow_arm_left.joint_pos)
                obs['qvel'] = np.array(follow_arm_left.joint_vel)
                obs['effort'] = np.array(follow_arm_left.joint_cur)
                obs['eef'] = follow_arm_left_eef_array

            if self.opt.use_chassis:
                obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            else:
                obs['base_vel'] = [0.0, 0.0]

            # 第一帧 只包含first， fisrt只保存StepType.FIRST
            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                timesteps.append(ts)
                continue

            # 时间步
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)

            # 主臂保存状态
            master_arm_left_eef_array = [master_arm_left_eef.x, master_arm_left_eef.y, master_arm_left_eef.z,
                                          master_arm_left_eef.roll, master_arm_left_eef.pitch, master_arm_left_eef.yaw, master_arm_left_eef.gripper]
            if not self.opt.use_single_arm:
                action = np.concatenate((np.array(master_arm_left.joint_pos),
                                        np.array(master_arm_right.joint_pos)), axis=0)
                master_arm_right_eef_array = [master_arm_right_eef.x, master_arm_right_eef.y, master_arm_right_eef.z,
                                           master_arm_right_eef.roll, master_arm_right_eef.pitch, master_arm_right_eef.yaw, master_arm_right_eef.gripper]
                action_eef = np.concatenate((master_arm_left_eef_array, master_arm_right_eef_array),axis=0)
            else:
                action = np.array(master_arm_left.joint_pos)
                action_eef = master_arm_left_eef_array
                
            actions.append(action)
            actions_eef.append(action_eef)
            
            timesteps.append(ts)
            print("Frame data: ", count)
            if rospy.is_shutdown():
                exit(-1)
            rate.sleep()

        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))

        return timesteps, actions, actions_eef


def main(opt):
    config = load_yaml(opt.data)
    ros_operator = RosOperator(opt, config)
    timesteps, actions, actions_eef = ros_operator.process()

    if(len(actions) < opt.max_timesteps):
        print("\033[31m\nSave failure, please record %s timesteps of data.\033[0m\n" %opt.max_timesteps)
        exit(-1)

    if not os.path.exists(opt.datasets):
        os.makedirs(opt.datasets)
    dataset_path = os.path.join(opt.datasets, "episode_" + str(opt.episode_idx))
    save_data(opt, timesteps, actions, actions_eef, dataset_path, opt.use_single_arm)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets', type=str, default="./datasets", help='dataset dir')
    parser.add_argument('--episode_idx', type=int, default=0, help='episode index')
    parser.add_argument('--max_timesteps', type=int, default=600, help='max timesteps')
    parser.add_argument('--frame_rate', type=int, default=90, help='frame rate')

    parser.add_argument('--data', type=str, default="./data/config.yaml", help='config file')

    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['cam_left_wrist', 'cam_right_wrist', 'cam_head', ],
                        default=['cam_left_wrist'], help='camera names')

    parser.add_argument('--use_chassis', action='store_true', help='use robot base')
    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')

    
    parser.add_argument('--is_compress', action='store_true', help='compress image') # 是否压缩图像
    parser.add_argument('--use_single_arm', action='store_true', help='if use single arm') # 是否使用单臂遥操作（一个主臂一个从臂）
    
    

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
