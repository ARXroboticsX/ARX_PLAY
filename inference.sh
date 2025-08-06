#!/bin/bash

workspace=$(dirname "$(pwd)")

max_publish_step=10000
ckpt_dir=weights
ckpt_name=policy_best.ckpt

check_path() {
    if [ ! -d "$1" ]; then
        echo "Directory not found: $1"

        exit 1
    fi
}

check_executable() {
    if [ ! -x "$1" ]; then
        echo "Script not executable: $1"
        exit 1
    fi
}

check_path "${workspace}/ARX_PLAY/realsense_camera"
check_path "${workspace}/ARX_PLAY/mobile_aloha"

check_executable "${workspace}/ARX_PLAY/realsense_camera/realsense.sh"

ACTIVATE_CMD=""
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda info --envs | grep -qE "^act[[:space:]]"; then
        ACTIVATE_CMD="conda activate act"
    fi
fi

if [ -z "$ACTIVATE_CMD" ]; then
    if [ -f "mobile_aloha/venv/bin/activate" ]; then
        ACTIVATE_CMD="source ./venv/bin/activate"
    else
        echo -e "\033[31m未找到运行环境\033[0m"
        exit 1
    fi
fi

gnome-terminal --title="realsense" -- bash -c "cd ${workspace}/ARX_PLAY/realsense_camera; bash realsense.sh; exec bash"
sleep 1
gnome-terminal --title="inference" -- bash -c "cd ${workspace}/ARX_PLAY/mobile_aloha; $ACTIVATE_CMD; python inference.py --max_publish_step $max_publish_step --ckpt_dir $ckpt_dir --ckpt_name $ckpt_name; exec bash"
sleep 1