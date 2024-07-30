# coding=utf-8
import os
import numpy as np
import cv2
import h5py
import argparse
import yaml
import rospy

from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist

import sys

sys.path.append("./")

is_compressed = False


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def load_hdf5(dataset_name):
    global is_compressed

    dataset_path = dataset_name
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


def main(opt):
    rospy.init_node("replay_node")
    bridge = CvBridge()

    config = load_yaml(opt.data)

    img_left_publisher = rospy.Publisher(config['camera_config']['original_image']['img_left_topic'],
                                         Image, queue_size=10)
    img_right_publisher = rospy.Publisher(config['camera_config']['original_image']['img_right_topic'],
                                          Image, queue_size=10)
    img_head_publisher = rospy.Publisher(config['camera_config']['original_image']['img_head_topic'],
                                          Image, queue_size=10)

    master_arm_left_publisher = rospy.Publisher(config['arm_config']['master_arm_left_topic'],
                                                JointState, queue_size=10)
    master_arm_right_publisher = rospy.Publisher(config['arm_config']['master_arm_right_topic'],
                                                 JointState, queue_size=10)
    follow_arm_left_publisher = rospy.Publisher(config['arm_config']['follow_arm_left_topic'],
                                                JointState, queue_size=10)
    follow_arm_right_publisher = rospy.Publisher(config['arm_config']['follow_arm_right_topic'],
                                                 JointState, queue_size=10)

    robot_base_publisher = rospy.Publisher(config['base_config']['robot_base_cmd_topic'],
                                           Twist, queue_size=10)

    origin_left = [-0.0057, -0.031, -0.0122, -0.032, 0.0099, 0.0179, 0.2279]
    origin_right = [0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]

    joint_state_msg = JointState()
    joint_state_msg.header = Header()
    joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
    twist_msg = Twist()

    rate = rospy.Rate(opt.frame_rate)

    qposs, qvels, efforts, actions, base_actions, image_dicts = load_hdf5(opt.episode)

    if opt.only_pub_master:
        last_action = [-0.0057, -0.031, -0.0122, -0.032, 0.0099, 0.0179, 0.2279,
                       0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]

        rate = rospy.Rate(100)
        for action in actions:
            if (rospy.is_shutdown()):
                break

            new_actions = np.linspace(last_action, action, 20)  # 插值
            last_action = action
            for act in new_actions:
                print(np.round(act[:7], 4))
                cur_timestamp = rospy.Time.now()  # 设置时间戳
                joint_state_msg.header.stamp = cur_timestamp

                joint_state_msg.position = act[:7]
                master_arm_left_publisher.publish(joint_state_msg)

                joint_state_msg.position = act[7:]
                master_arm_right_publisher.publish(joint_state_msg)

                if (rospy.is_shutdown()):
                    break
                rate.sleep()

    else:
        i = 0
        while (not rospy.is_shutdown() and i < len(actions)):
            print("left: ", np.round(qposs[i][:7], 4), " right: ", np.round(qposs[i][7:], 4))

            cam_names = [k for k in image_dicts.keys()]
            image0 = image_dicts[cam_names[0]][i]
            image1 = image_dicts[cam_names[1]][i]
            image2 = image_dicts[cam_names[2]][i]

            if is_compressed == False:
                # swap B and R channel
                image0 = image0[:, :, [2, 1, 0]]
                image1 = image1[:, :, [2, 1, 0]]
                image2 = image2[:, :, [2, 1, 0]]

            cur_timestamp = rospy.Time.now()  # 设置时间戳

            joint_state_msg.header.stamp = cur_timestamp
            joint_state_msg.position = actions[i][:7]
            master_arm_left_publisher.publish(joint_state_msg)

            joint_state_msg.position = actions[i][7:]
            master_arm_right_publisher.publish(joint_state_msg)

            joint_state_msg.position = qposs[i][:7]
            follow_arm_left_publisher.publish(joint_state_msg)

            joint_state_msg.position = qposs[i][7:]
            follow_arm_right_publisher.publish(joint_state_msg)

            img_head_publisher.publish(bridge.cv2_to_imgmsg(image0, "bgr8"))
            img_left_publisher.publish(bridge.cv2_to_imgmsg(image1, "bgr8"))
            img_right_publisher.publish(bridge.cv2_to_imgmsg(image2, "bgr8"))

            twist_msg.linear.x = base_actions[i][0]
            twist_msg.angular.z = base_actions[i][1]

            if opt.use_robot_base:
                robot_base_publisher.publish(twist_msg)

            i += 1
            rate.sleep()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--episode', type=str, help='episode', required=True)

    parser.add_argument('--frame_rate', type=int, default=90, help='frame rate')
    parser.add_argument('--data', type=str, default="./data/config.yaml", help='config file')

    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                        help='camera names')

    parser.add_argument('--use_robot_base', action='store_true', help='use robot base')
    parser.add_argument('--robot_base_topic', type=str, default='/cmd_vel', help='robot base topic')

    parser.add_argument('--only_pub_master', action='store_true', help='only pub master')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
