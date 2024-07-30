#!/bin/bash

# 启动相机
gnome-terminal -t "realsense" -x bash -c "source ./devel/setup.bash;roslaunch realsense2_camera rs_multiple_devices.launch;"