import copy
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Models.AbsolutePositionalEncoding import SinPositionalEncoding, APE2D, LPE2D
from Models.AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from Models.Attention import *


def Encoder_factory(config):
    if config['Training_mode'] == 'Rep-Learning':
        model = EEG2Rep(config, num_classes=config['num_labels'])
    else:
        model = EEG_JEPA_Sup(config, num_classes=config['num_labels'])
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        d_model = config['emb_size']
        attn_heads = config['num_heads']
        # d_ffn = 4 * d_model
        d_ffn = config['dim_ff']
        layers = config['layers']
        dropout = config['dropout']
        enable_res_parameter = True
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x


class Target_Encoder(nn.Module):
    def __init__(self, config):
        super(Target_Encoder, self).__init__()
        d_model = config['emb_size']
        attn_heads = config['num_heads']
        # d_ffn = 4 * d_model
        d_ffn = config['dim_ff']
        layers = config['layers']
        dropout = config['dropout']
        enable_res_parameter = True
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

    def forward(self, x):
        layer_outputs = []
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
            layer_outputs.append(x.clone())

        # Return the average of the last three layer outputs
        avg_last_3_layers = [F.instance_norm(tl.float()) for tl in layer_outputs]
        avg_last_3_layers = sum(avg_last_3_layers) / len(avg_last_3_layers)
        return avg_last_3_layers


class EEG2Rep(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        """
         channel_size: number of EEG channels
         seq_len: number of timepoints in a window
        """
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']  # d_x
        # Embedding Layer -----------------------------------------------------------
        config['pooling_size'] = 5  # Max pooling size in input embedding
        seq_len = int(seq_len / config['pooling_size'])
        self.PatchEmbedding = InputEmbedding(config)
        self.PositionalEncoding = PositionalEmbedding(seq_len, emb_size)
        # -------------------------------------------------------------------------

        self.momentum = config['momentum']
        self.linear_proba = True
        self.device = config['device']
        self.mask_ratio = config['mask_ratio']
        self.mask_len = int(config['mask_ratio'] * seq_len)
        self.mask_token = nn.Parameter(torch.randn(emb_size, ))
        self.contex_encoder = Encoder(config)
        self.target_encoder = copy.deepcopy(self.contex_encoder)
        # self.target_encoder = Target_Encoder(config)
        self.Predictor = Predictor(emb_size, config['num_heads'], config['dim_ff'], 1, config['pre_layers'])
        self.predict_head = nn.Linear(emb_size, config['num_labels'])

        self.gap = nn.AdaptiveAvgPool1d(1)

    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def linear_prob(self, x):
        with (torch.no_grad()):
            patches = self.PatchEmbedding(x)
            patches += self.PositionalEncoding(patches)
            out = self.contex_encoder(patches)
            out = out.transpose(2, 1)
            out = self.gap(out)
            return out.squeeze()

    def pretrain_forward(self, x):

        patches = self.PatchEmbedding(x)
        patches += self.PositionalEncoding(patches)

        rep_mask_token = self.mask_token.repeat(patches.shape[0], patches.shape[1], 1)
        rep_mask_token = + self.PositionalEncoding(rep_mask_token)

        index = np.arange(patches.shape[1])
        m_index_chunk = select_chunks_by_indices(index, chunk_count=2, target_percentage=self.mask_ratio)
        m_index = np.ravel(m_index_chunk)
        v_index = np.setdiff1d(index, m_index)

        # v_index = np.sort(np.unique(np.concatenate(m_index_chunk)))
        # m_index = np.setdiff1d(index, v_index)

        # random.shuffle(index)
        # v_index = index[:-self.mask_len]
        # m_index = index[-self.mask_len:]

        visible = patches[:, v_index, :]
        rep_mask_token = rep_mask_token[:, m_index, :]
        rep_contex = self.contex_encoder(visible)
        with torch.no_grad():
            rep_target = self.target_encoder(patches)
            rep_mask = rep_target[:, m_index, :]
        rep_mask_prediction = self.Predictor(rep_contex, rep_mask_token)
        return [rep_mask, rep_mask_prediction, rep_contex, rep_target]

    def forward(self, x):

        patches = self.PatchEmbedding(x)
        patches += self.PositionalEncoding(patches)
        out = self.contex_encoder(patches)
        return self.predict_head(torch.mean(out, dim=1))


class InputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']  # d_x (input embedding dimension)
        k = 40
        # Embedding Layer -----------------------------------------------------------
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(channel_size, 1))
        self.spatial_padding = nn.ReflectionPad2d((int(np.floor((k - 1) / 2)), int(np.ceil((k - 1) / 2)), 0, 0))
        self.spatialwise_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))
        self.spatialwise_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))
        self.SiLU = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, config['pooling_size']), stride=(1, config['pooling_size']))

    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.depthwise_conv(out)  # (bs, embedding, 1 , T)
        out = out.transpose(1, 2)  # (bs, 1, embedding, T)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv1(out)  # (bs, 1, embedding, T)
        out = self.SiLU(out)
        out = self.maxpool(out)  # (bs, 1, embedding, T // m)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv2(out)
        out = out.squeeze(1)  # (bs, embedding, T // m)
        out = out.transpose(1, 2)  # (bs, T // m, embedding)
        patches = self.SiLU(out)
        return patches


def select_chunks_by_indices(time_step_indices, chunk_count=2, target_percentage=0.5):
    # Get the total number of time steps
    total_time_steps = len(time_step_indices)

    # Calculate the desired total time steps for the selected chunks
    target_total_time_steps = int(total_time_steps * target_percentage)

    # Calculate the size of each chunk
    chunk_size = target_total_time_steps // chunk_count

    # Calculate the minimum distance between starting points
    min_starting_distance = chunk_size

    # Randomly select starting points for each chunk with minimum distance
    start_points = [random.randint(0, total_time_steps - chunk_size)]
    # Randomly select starting points for each subsequent chunk with minimum distance
    for _ in range(chunk_count - 1):
        next_start_point = random.randint(0, total_time_steps - chunk_size)
        start_points.append(next_start_point)

    # Select non-overlapping chunks using indices
    selected_chunks_indices = [time_step_indices[start:start + chunk_size] for start in start_points]

    return selected_chunks_indices


class EEG_JEPA_Sup(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        """
         channel_size: number of EEG channels
         seq_len: number of timepoints in a window
         k: length of spatial filters (i.e. how much you look in time)
         m: maxpool size
        """
        # Parameters Initialization -----------------------------------------------
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']

        # Embedding Layer -----------------------------------------------------------
        m = 5
        seq_len = int(seq_len / m)
        self.PatchEmbedding = PatchEmbedding(config)

        self.SinPositionalEncoding = PositionalEmbedding(seq_len, emb_size)
        self.contex_encoder = Encoder(config)
        # self.predict_head = nn.Linear(emb_size*int(seq_len/m), config['num_labels'])
        self.predict_head = nn.Linear(emb_size, config['num_labels'])

    def forward(self, x):
        patches = self.PatchEmbedding(x)
        patches += self.SinPositionalEncoding(patches)
        out = self.contex_encoder(patches)
        return self.predict_head(torch.mean(out, dim=1))
        # out = out.view(out.shape[0], -1)
        # return self.predict_head(out)


class InputProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        self.Input_projection = nn.Conv1d(channel_size, emb_size, kernel_size=config['patch_size'],
                                          stride=config['patch_size'])

    def forward(self, x):
        patches = self.Input_projection(x).transpose(1, 2)
        return patches


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x):
        return super(CausalConv1d, self).forward(nn.functional.pad(x, (self.__padding, 0)))


class Predictor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Predictor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token


def reshape_4d_to_3d(input_tensor):
    # Get the current dimensions
    B, C, L, P = input_tensor.size()
    # Reshape the tensor to 3D
    output_tensor = input_tensor.transpose(1, 2).reshape(B, C * L, P)
    return output_tensor


def return_4d_index(input_tensor, m_index):
    # Get the current dimensions
    B, C, L, P = input_tensor.size()
    indices = []
    for channel_index in m_index:
        base_index = channel_index * L
        indices.extend(np.arange(base_index, base_index + L))
    return indices


class PatchingModule(nn.Module):
    def __init__(self, kernel_size, stride):
        super(PatchingModule, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        patches = x.unfold(2, self.kernel_size, self.stride)
        return patches


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.gelu(self.norm(self.conv1(x)))
        out = self.gelu(self.norm(self.conv2(out)))
        if out.shape[1] == residual.shape[1]:
            out += residual
        return out


class GLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GLUBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, 2 * out_channels, kernel_size=3, stride=1, padding=1)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        out = self.glu(self.conv(x))
        return out


class MyModel(nn.Module):
    def __init__(self, config, num_classes, num_blocks=5):
        super(MyModel, self).__init__()
        self.blocks = nn.ModuleList()
        input_dim, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        for k in range(num_blocks):
            in_channels = input_dim if k == 0 else 320
            out_channels = 320
            dilation = 2 ** (2 * k % 5)

            self.blocks.append(ResidualBlock(in_channels, out_channels, dilation))
            self.blocks.append(GLUBlock(out_channels, out_channels))
            input_dim = out_channels
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.C_out = nn.Linear(out_channels, num_classes)

    def forward(self, x):

        for block in self.blocks:
            x = block(x)
        x = self.gap(x)
        x = self.C_out(x.squeeze(-1))
        return x
