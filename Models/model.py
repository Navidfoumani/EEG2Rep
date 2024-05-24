import copy
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Models.Attention import *


def Encoder_factory(config):
    model = EEG2Rep(config, num_classes=config['num_labels'])
    return model


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
        config['pooling_size'] = 2  # Max pooling size in input embedding
        seq_len = int(seq_len / config['pooling_size'])  # Number of patches (l)
        self.InputEmbedding = InputEmbedding(config)  # input (Batch,Channel, length) -> output (Batch, l, d_x)
        self.PositionalEncoding = PositionalEmbedding(seq_len, emb_size)
        # -------------------------------------------------------------------------
        self.momentum = config['momentum']
        self.device = config['device']
        self.mask_ratio = config['mask_ratio']
        self.mask_len = int(config['mask_ratio'] * seq_len)
        self.mask_token = nn.Parameter(torch.randn(emb_size, ))
        self.contex_encoder = Encoder(config)
        self.target_encoder = copy.deepcopy(self.contex_encoder)
        self.Predictor = Predictor(emb_size, config['num_heads'], config['dim_ff'], 1, config['pre_layers'])
        self.predict_head = nn.Linear(emb_size, config['num_labels'])
        self.Norm = nn.LayerNorm(emb_size)
        self.Norm2 = nn.LayerNorm(emb_size)
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
            patches = self.InputEmbedding(x)
            patches = self.Norm(patches)
            patches = patches + self.PositionalEncoding(patches)
            patches = self.Norm2(patches)
            out = self.contex_encoder(patches)
            out = out.transpose(2, 1)
            out = self.gap(out)
            return out.squeeze()

    def pretrain_forward(self, x):
        patches = self.InputEmbedding(x)  # (Batch, l, d_x)
        patches = self.Norm(patches)
        patches = patches + self.PositionalEncoding(patches)
        patches = self.Norm2(patches)

        rep_mask_token = self.mask_token.repeat(patches.shape[0], patches.shape[1], 1)
        rep_mask_token = rep_mask_token + self.PositionalEncoding(rep_mask_token)

        index = np.arange(patches.shape[1])
        index_chunk = Semantic_Subsequence_Preserving(index, 2, self.mask_ratio)
        v_index = np.ravel(index_chunk)
        m_index = np.setdiff1d(index, v_index)

        visible = patches[:, v_index, :]
        rep_mask_token = rep_mask_token[:, m_index, :]
        rep_contex = self.contex_encoder(visible)
        with torch.no_grad():
            rep_target = self.target_encoder(patches)
            rep_mask = rep_target[:, m_index, :]
        rep_mask_prediction = self.Predictor(rep_contex, rep_mask_token)
        return [rep_mask, rep_mask_prediction, rep_contex, rep_target]

    def forward(self, x):
        patches = self.InputEmbedding(x)
        patches = self.Norm(patches)
        patches = patches + self.PositionalEncoding(patches)
        patches = self.Norm2(patches)
        out = self.contex_encoder(patches)
        return self.predict_head(torch.mean(out, dim=1))


class InputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']  # d_x (input embedding dimension)
        k = 7
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


def Semantic_Subsequence_Preserving(time_step_indices, chunk_count, target_percentage):
    # Get the total number of time steps
    total_time_steps = len(time_step_indices)
    # Calculate the desired total time steps for the selected chunks
    target_total_time_steps = int(total_time_steps * target_percentage)

    # Calculate the size of each chunk
    chunk_size = target_total_time_steps // chunk_count

    # Randomly select starting points for each chunk with minimum distance
    start_points = [random.randint(0, total_time_steps - chunk_size)]
    # Randomly select starting points for each subsequent chunk with minimum distance
    for _ in range(chunk_count - 1):
        next_start_point = random.randint(0, total_time_steps - chunk_size)
        start_points.append(next_start_point)

    # Select non-overlapping chunks using indices
    selected_chunks_indices = [time_step_indices[start:start + chunk_size] for start in start_points]

    return selected_chunks_indices


class Predictor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Predictor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


