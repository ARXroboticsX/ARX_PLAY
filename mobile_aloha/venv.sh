#!/bin/bash

sudo apt install -y python3.8
sudo apt install -y python3-pip python3-venv

python3.8 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cd detr
pip install -e .