# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone, DepthNet
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np
from collections import OrderedDict
import sys

sys.path.append("./")
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import IPython

e = IPython.embed

from utils.utils import get_gpu_mem_info


def print_gpu_mem():
    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info()
    return (gpu_mem_used / gpu_mem_total) * 100


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbones, depth_backbones, transformer, encoder, policy_config):

        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            states_dim: robot state dimension of the environment
            chunk_size: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.chunk_size = policy_config.chunk_size
        self.camera_names = policy_config.camera_names
        self.states_dim = int(policy_config.states_dim)

        self.action_dim = policy_config.action_dim  # 冗余输出
        # print(f'{self.action_dim=}')

        self.kl_weight = policy_config.kl_weight
        self.use_base = policy_config.use_base

        self.transformer = transformer
        self.encoder = encoder
        self.hidden_dim = transformer.d_model
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.query_embed = nn.Embedding(self.chunk_size, self.hidden_dim)

        # input_dim = 14 + 7 # robot_state + env_state
        self.input_proj_left_state = nn.Linear(self.states_dim, self.hidden_dim)
        self.input_proj_right_state = nn.Linear(self.states_dim, self.hidden_dim)
        self.input_proj_robot_base = nn.Linear(3, self.hidden_dim)
        self.input_proj_robot_head = nn.Linear(3, self.hidden_dim)
        self.input_proj_base_velocity = nn.Linear(4, self.hidden_dim)

        if backbones is not None:
            # print("backbones[0]", backbones[0])
            if depth_backbones is not None:
                self.depth_backbones = nn.ModuleList(depth_backbones)
                self.input_proj = nn.Conv2d(backbones[0].num_channels + depth_backbones[0].num_channels,
                                            self.hidden_dim,
                                            kernel_size=1)
            else:
                self.depth_backbones = None
                self.input_proj = nn.Conv2d(backbones[0].num_channels, self.hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)

        else:
            self.pos = torch.nn.Embedding(2, self.hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, self.hidden_dim)  # extra cls token embedding

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, self.hidden_dim)  # project latent sample to embedding
        self.latent_pos = nn.Embedding(1, self.hidden_dim)

        pos_embed_dim = 1
        pos_embed_dim = pos_embed_dim + 3 if self.use_base else pos_embed_dim

        self.robot_state_pos = nn.Embedding(pos_embed_dim, self.hidden_dim)

        self.encoder_action_proj = nn.Linear(self.action_dim, self.hidden_dim)  # project action to embedding
        self.encoder_left_states_proj = nn.Linear(self.states_dim, self.hidden_dim)  # project qpos to embedding
        self.encoder_right_states_proj = nn.Linear(self.states_dim, self.hidden_dim)  # project qpos to embedding
        self.encoder_robot_base_proj = nn.Linear(3, self.hidden_dim)  # project qpos to embedding
        self.encoder_robot_head_proj = nn.Linear(3, self.hidden_dim)  # project qpos to embedding
        self.encoder_base_velocity_proj = nn.Linear(4, self.hidden_dim)

        self.latent_proj = nn.Linear(self.hidden_dim, self.latent_dim * 2)  # project hidden state to latent std, var

        self.encoder_addition_block_dim = 2  # cls + joints
        self.encoder_addition_block_dim = self.encoder_addition_block_dim + 3 if self.use_base else self.encoder_addition_block_dim

        self.register_buffer('pos_table', get_sinusoid_encoding_table(self.encoder_addition_block_dim + self.chunk_size,
                                                                      self.hidden_dim))  # cls

    def encode_process(self, left_states, right_states, robot_base=None, robot_head=None, base_velocity=None,
                       actions=None, action_is_pad=None):
        bs = left_states.shape[0]
        is_training = actions is not None  # train or val

        # 將states拆分成两部分输入
        # print("\n图像编码器 start1 :", print_gpu_mem())

        if is_training:  # self.hidden_dim输入参数是512
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, self.hidden_dim)

            left_states_embed = self.encoder_left_states_proj(left_states)  # (bs, self.hidden_dim)
            left_states_embed = torch.unsqueeze(left_states_embed, dim=1)

            cls_embed = self.cls_embed.weight  # (1, self.hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, dim=0).repeat(bs, 1, 1)  # (bs, 1, self.hidden_dim)

            if self.use_base:
                robot_base_embed = self.encoder_robot_base_proj(robot_base)  # (bs, self.hidden_dim)
                robot_base_embed = torch.unsqueeze(robot_base_embed, dim=1)

                robot_head_embed = self.encoder_robot_head_proj(robot_head)  # (bs, self.hidden_dim)
                robot_head_embed = torch.unsqueeze(robot_head_embed, dim=1)

                base_velocity_embed = self.encoder_base_velocity_proj(base_velocity)  # (bs, self.hidden_dim)
                base_velocity_embed = torch.unsqueeze(base_velocity_embed, dim=1)

            if self.use_base:
                encoder_input = torch.cat(
                    [cls_embed, left_states_embed, robot_base_embed, robot_head_embed, base_velocity_embed,
                     action_embed],
                    dim=1)  # (bs, seq+1, self.hidden_dim)
            else:
                encoder_input = torch.cat([cls_embed, left_states_embed, action_embed],
                                          dim=1)  # (bs, seq+1, self.hidden_dim)

            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, self.hidden_dim)
            cls_joint_is_pad = torch.zeros((bs, self.encoder_addition_block_dim),
                                           dtype=torch.bool)  # cls+left_states + right_states=(8, 3)
            cls_joint_is_pad = cls_joint_is_pad.to(left_states.device)

            is_pad = torch.cat([cls_joint_is_pad, action_is_pad], dim=1)  # (bs, seq+1)

            # obtain position embedding  合并位置编码
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, self.hidden_dim)

            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # take cls output only

            # 线性层  self.hidden_dim扩大到64
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(left_states.device)
            latent_input = self.latent_out_proj(latent_sample)

        return latent_input, mu, logvar

    def forward(self, image, depth_image, left_states, right_states, robot_base=None, robot_head=None,
                base_velocity=None, actions=None, action_is_pad=None, command_embedding=None):
        latent_input, mu, logvar = self.encode_process(left_states, right_states,
                                                       robot_base=robot_base, robot_head=robot_head,
                                                       base_velocity=base_velocity,
                                                       actions=actions, action_is_pad=action_is_pad)

        # print("forward: ", qpos.shape, image.shape, env_state, actions.shape, action_is_pad.shape)

        is_training = actions is not None  # train or val

        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_depth_features = []
        all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            # features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
            features, img_src_pos = self.backbones[cam_id](image[:, cam_id])  # HARDCODED
            features = features[0]  # take the last layer feature
            img_src_pos = img_src_pos[0]
            if self.depth_backbones is not None and depth_image is not None:
                features_depth = self.depth_backbones[cam_id](depth_image[:, cam_id].unsqueeze(dim=1))
                all_cam_features.append(self.input_proj(torch.cat([features, features_depth], dim=1)))
            else:
                all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(img_src_pos)

        left_states_input = self.input_proj_left_state(left_states)
        left_states_input = torch.unsqueeze(left_states_input, dim=0)

        # fold camera dimension into width dimension
        img_src = torch.cat(all_cam_features, dim=3)
        img_src_pos = torch.cat(all_cam_pos, dim=3)

        latent_input = torch.unsqueeze(latent_input, dim=0)

        if self.use_base:
            robot_base_input = self.input_proj_robot_base(robot_base)
            robot_base_input = torch.unsqueeze(robot_base_input, dim=0)

            robot_head_input = self.input_proj_robot_head(robot_head)
            robot_head_input = torch.unsqueeze(robot_head_input, dim=0)

            robot_velocity_input = self.input_proj_base_velocity(base_velocity)
            robot_velocity_input = torch.unsqueeze(robot_velocity_input, dim=0)
        else:
            robot_base_input = None
            robot_head_input = None
            robot_velocity_input = None

        right_states_input = None

        hs = self.transformer(self.query_embed.weight,
                              img_src, img_src_pos, None,
                              left_states_input, right_states_input,
                              robot_base_input, robot_head_input, robot_velocity_input,
                              self.robot_state_pos.weight, latent_input, self.latent_pos.weight)[0]

        a_hat = self.action_head(hs)

        return a_hat, [mu, logvar]


class CNNMLP(nn.Module):
    def __init__(self, backbones, depth_backbones, states_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            states_dim: robot state dimension of the environment
            chunk_size: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.depth_backbones = depth_backbones
        # print(f'^^^^^^^^^^^^^^^^^^^^^{states_dim=}')
        self.action_head = nn.Linear(1000, states_dim)  # TODO add more

        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []

            for i, backbone in enumerate(backbones):
                num_channels = backbone.num_channels
                if self.depth_backbones is not None:
                    num_channels += depth_backbones[i].num_channels
                down_proj = nn.Sequential(
                    nn.Conv2d(num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + states_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=states_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, image, depth_image, robot_state, actions=None, action_is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        bs, _ = robot_state.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]  # take the last layer feature
            if self.depth_backbones is not None and depth_image is not None:
                features_depth = self.depth_backbones[cam_id](depth_image[:, cam_id].unsqueeze(dim=1))
                all_cam_features.append(self.backbone_down_projs[cam_id](torch.cat([features, features_depth], axis=1)))
            else:
                all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)  # 768 each
        features = torch.cat([flattened_features, robot_state], axis=1)  # qpos: 14
        a_hat = self.mlp(features)
        # print(f')****************a_hat.shape={a_hat.shape=}')
        return a_hat


class Diffusion(nn.Module):
    def __init__(self, backbones, pools, linears, depth_backbones, states_dim, chunk_size,
                 observation_horizon, action_horizon, num_inference_timesteps,
                 ema_power, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            states_dim: robot state dimension of the environment
            chunk_size: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.backbones = nn.ModuleList(backbones)
        self.backbones = replace_bn_with_gn(self.backbones)  # TODO
        self.pools = nn.ModuleList(pools)
        self.linears = nn.ModuleList(linears)
        self.depth_backbones = depth_backbones
        if depth_backbones is not None:
            self.depth_backbones = nn.ModuleList(depth_backbones)
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.chunk_size = chunk_size
        self.num_inference_timesteps = num_inference_timesteps
        self.ema_power = ema_power
        self.states_dim = states_dim
        self.weight_decay = 0
        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = states_dim
        self.obs_dim = self.feature_dimension * len(self.camera_names) + states_dim  # camera features and proprio
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.states_dim,
            global_cond_dim=self.obs_dim * self.observation_horizon
        )
        if depth_backbones is not None:
            nets = nn.ModuleDict({
                'policy': nn.ModuleDict({
                    'backbones': self.backbones,
                    'depth_backbones': self.depth_backbones,
                    'pools': self.pools,
                    'linears': self.linears,
                    'noise_pred_net': self.noise_pred_net
                })
            })
        else:
            nets = nn.ModuleDict({
                'policy': nn.ModuleDict({
                    'backbones': self.backbones,
                    'pools': self.pools,
                    'linears': self.linears,
                    'noise_pred_net': self.noise_pred_net
                })
            })

        nets = nets.float().cuda()
        ENABLE_EMA = False  # True
        if ENABLE_EMA:
            ema = EMAModel(model=nets, power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

    def forward(self, image, depth_image, robot_state, actions=None, action_is_pad=None):
        B = robot_state.shape[0]
        if actions is not None:  # training time
            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                if depth_image is not None:
                    features_depth = self.depth_backbones[cam_id](depth_image[:, cam_id].unsqueeze(dim=1))
                    cam_features = torch.cat([cam_features, features_depth], axis=1)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)
            obs_cond = torch.cat(all_features + [robot_state], dim=1)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=obs_cond.device
            ).long()
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            # predict the noise residual
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
            if self.ema is not None:
                self.ema.step(nets)
            return noise, noise_pred
        else:
            To = self.observation_horizon
            Ta = self.action_horizon
            Tp = self.chunk_size

            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model

            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                if depth_image is not None:
                    features_depth = self.depth_backbones[cam_id](depth_image[:, cam_id].unsqueeze(dim=1))
                    cam_features = torch.cat([cam_features, features_depth], axis=1)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)
            obs_cond = torch.cat(all_features + [robot_state], dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, self.states_dim), device=obs_cond.device)
            naction = noisy_action
            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )
                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status


def build_diffusion(args):
    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    pools = []
    linears = []
    depth_backbones = None
    if args.use_depth_image:
        depth_backbones = []

    for _ in args.camera_names:
        backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))
        num_channels = 512

        if args.use_depth_image:
            depth_backbones.append(DepthNet())
            num_channels += depth_backbones[-1].num_channels

        pools.append(SpatialSoftmax(**{'input_shape': [num_channels, 15, 20], 'num_kp': 32, 'temperature': 1.0,
                                       'learnable_temperature': False, 'noise_std': 0.0}))
        linears.append(torch.nn.Linear(int(np.prod([32, 2])), 64))

    model = Diffusion(
        backbones,
        pools,
        linears,
        depth_backbones,
        states_dim=args.states_dim,
        chunk_size=args.chunk_size,
        observation_horizon=args.observation_horizon,
        action_horizon=args.action_horizon,
        num_inference_timesteps=args.num_inference_timesteps,
        ema_power=args.ema_power,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))
    return model


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim  # 256
    dropout = args.dropout  # 0.1
    nhead = args.nheads  # 15
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)

    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build_vae(args):
    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []  # 空的网络list
    depth_backbones = None
    if args.use_depth_image:
        depth_backbones = []

    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)
        # if args.use_depth_image:
        #     depth_backbones.append(DepthNet())

    transformer = build_transformer(args)  # 构建trans层

    # encoder = None
    encoder = build_encoder(args)  # 构建编码成和解码层

    model = DETRVAE(
        backbones,
        depth_backbones,
        transformer,
        encoder,
        args,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_cnnmlp(args):
    if args.use_base:
        args.states_dim = args.states_dim + 5

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []  # 空的网络list
    depth_backbones = None
    if args.use_depth_image:
        depth_backbones = []

    # backbone = build_backbone(args)  # 位置编码和主干网络组合成特征提取器
    # backbones.append(backbone)
    # if args.use_depth_image:
    #     depth_backbones.append(DepthNet())

    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)
        if args.use_depth_image:
            depth_backbones.append(DepthNet())

    model = CNNMLP(
        backbones,
        depth_backbones,
        states_dim=args.states_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model
