#!/bin/bash

workspace=$(dirname "$(pwd)")

datasets=datasets
ckpt_dir=weights
num_episodes=50
batch_size=32
epochs=3000

check_path() {
    if [ ! -d "$1" ]; then
        echo "Directory not found: $1"

        exit 1
    fi
}

check_path "${workspace}/ARX_PLAY/mobile_aloha"
check_path "${workspace}/ARX_PLAY/mobile_aloha/$datasets"

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

gnome-terminal --title="train" -- bash -c "cd ${workspace}/ARX_PLAY/mobile_aloha/; $ACTIVATE_CMD; python train.py --datasets $datasets --ckpt_dir $ckpt_dir --num_episodes $num_episodes --batch_size $batch_size --epochs $epochs; exec bash"
sleep 1