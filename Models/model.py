import random
import numpy as np
import torch
from torch import nn
from Models.AbsolutePositionalEncoding import SinPositionalEncoding, APE2D, LPE2D
from Models.AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from Models.Attention import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)


def Encoder_factory(config):
    if config['Training_mode'] == 'Rep-Learning':
        # model = EEG_JEPA(config, num_classes=config['num_labels'])
        model = Conv_EEG_JEPA(config, num_classes=config['num_labels'])
    else:
        model = EEG_JEPA_Sup(config, num_classes=config['num_labels'])
        # model = ConvTran(config, num_classes=config['num_labels'])
        # model = MyModel(config, num_classes=config['num_labels'])
        # model = EEG_FeatureExtractor(config, num_classes=config['num_labels'])
    return model


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        d_model = config['emb_size']
        attn_heads = config['num_heads']
        d_ffn = 4 * d_model
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


class EEG_JEPA(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        patch_size = config['patch_size']
        self.seq_len = int(seq_len/patch_size)
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=[1, 3], padding='same'),
                                          nn.BatchNorm2d(16),
                                          nn.GELU())

        self.embed_layer1_2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=[channel_size, 1], padding='valid'),
                                            nn.BatchNorm2d(16),
                                            nn.GELU())

        self.input_projection = nn.Conv1d(16, emb_size, kernel_size=patch_size, stride=patch_size)
        self.SinPositionalEncoding = PositionalEmbedding(self.seq_len, emb_size)

        self.momentum = config['momentum']
        self.linear_proba = True
        self.device = config['device']

        self.mask_len = int(config['mask_ratio'] * self.seq_len)
        self.mask_token = nn.Parameter(torch.randn(emb_size, ))
        self.contex_encoder = Encoder(config)
        self.target_encoder = Encoder(config)
        self.Predictor = Predictor(emb_size, num_heads, dim_ff, 1, config['pre_layers'])
        self.predict_head = nn.Linear(emb_size, config['num_labels'])


    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def linear_prob(self, x):
        with torch.no_grad():
            x = self.embed_layer1(x.unsqueeze(1))
            x = self.embed_layer1_2(x)
            patches = self.input_projection(x.squeeze(2)).transpose(1, 2)
            patches += self.SinPositionalEncoding(patches)
            out = self.target_encoder(patches)
            return out

    def pretrain_forward(self, x):

        x = self.embed_layer1(x.unsqueeze(1))
        x = self.embed_layer1_2(x)
        patches = self.input_projection(x.squeeze(2)).transpose(1, 2)
        patches += self.SinPositionalEncoding(patches)

        rep_mask_token = self.mask_token.repeat(patches.shape[0], patches.shape[1], 1) + self.SinPositionalEncoding(patches)

        index = np.arange(patches.shape[1])
        random.shuffle(index)
        v_index = index[:-self.mask_len]
        m_index = index[-self.mask_len:]
        visible = patches[:, v_index, :]
        rep_mask_token = rep_mask_token[:, m_index, :]
        rep_contex = self.contex_encoder(visible)
        with torch.no_grad():
            rep_target = self.target_encoder(patches)
            rep_mask = rep_target[:, m_index, :]
        rep_mask_prediction = self.Predictor(rep_contex, rep_mask_token)
        return [rep_mask, rep_mask_prediction, rep_contex, rep_target]

    def forward(self, x):
        x = self.embed_layer1(x.unsqueeze(1))
        x = self.embed_layer1_2(x)
        patches = self.input_projection(x.squeeze(2)).transpose(1, 2)
        patches += self.SinPositionalEncoding(patches)

        out = self.target_encoder(patches)
        return self.predict_head(torch.mean(out, dim=1))


class Conv_EEG_JEPA(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        """
         channel_size: number of EEG channels
         seq_len: number of timepoints in a window
         k: length of spatial filters (i.e. how much you look in time)
         m: maxpool size
        """
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        k = 40
        m = 5
        # Embedding Layer -----------------------------------------------------------
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(channel_size, 1))
        self.spatial_padding = nn.ReflectionPad2d((int(np.floor((k - 1) / 2)), int(np.ceil((k - 1) / 2)), 0, 0))
        self.spatialwise_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))
        self.spatialwise_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))
        self.SiLU = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, m), stride=(1, m))

        self.SinPositionalEncoding = PositionalEmbedding(int(seq_len / m), emb_size)

        self.momentum = config['momentum']
        self.linear_proba = True
        self.device = config['device']

        self.mask_len = int(config['mask_ratio'] * int(seq_len / m))
        self.mask_token = nn.Parameter(torch.randn(emb_size, ))
        self.contex_encoder = Encoder(config)
        self.target_encoder = Encoder(config)
        self.Predictor = Predictor(emb_size, config['num_heads'], config['dim_ff'], 1, config['pre_layers'])
        self.predict_head = nn.Linear(emb_size, config['num_labels'])


    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def linear_prob(self, x):
        with torch.no_grad():
            out = x.unsqueeze(1)
            out = self.depthwise_conv(out)  # (bs, embedding, 1 , T)
            out = out.permute(0, 2, 1, 3)  # (bs, 1, embedding, T)
            out = self.spatial_padding(out)
            out = self.spatialwise_conv1(out)  # (bs, 1, embedding, T)
            out = self.SiLU(out)
            out = self.maxpool(out)  # (bs, 1, embedding, T // m)
            out = self.spatial_padding(out)
            out = self.spatialwise_conv2(out)
            out = out.squeeze(1)  # (bs, embedding, T // m)
            out = out.permute(0, 2, 1)  # (bs, T // m, embedding)
            patches = self.SiLU(out)
            patches += self.SinPositionalEncoding(patches)
            out = self.contex_encoder(patches)
            return out

    def pretrain_forward(self, x):

        out = x.unsqueeze(1)
        out = self.depthwise_conv(out)  # (bs, embedding, 1 , T)
        out = out.permute(0, 2, 1, 3)  # (bs, 1, embedding, T)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv1(out)  # (bs, 1, embedding, T)
        out = self.SiLU(out)
        out = self.maxpool(out)  # (bs, 1, embedding, T // m)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv2(out)
        out = out.squeeze(1)  # (bs, embedding, T // m)
        out = out.permute(0, 2, 1)  # (bs, T // m, embedding)
        patches = self.SiLU(out)
        patches += self.SinPositionalEncoding(patches)

        rep_mask_token = self.mask_token.repeat(patches.shape[0], patches.shape[1], 1) + self.SinPositionalEncoding(patches)

        index = np.arange(patches.shape[1])
        random.shuffle(index)
        v_index = index[:-self.mask_len]
        m_index = index[-self.mask_len:]
        visible = patches[:, v_index, :]
        rep_mask_token = rep_mask_token[:, m_index, :]
        rep_contex = self.contex_encoder(visible)
        with torch.no_grad():
            rep_target = self.target_encoder(patches)
            rep_mask = rep_target[:, m_index, :]
        rep_mask_prediction = self.Predictor(rep_contex, rep_mask_token)
        return [rep_mask, rep_mask_prediction, rep_contex, rep_target]

    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.depthwise_conv(out)  # (bs, embedding, 1 , T)
        out = out.permute(0, 2, 1, 3)  # (bs, 1, embedding, T)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv1(out)  # (bs, 1, embedding, T)
        out = self.SiLU(out)
        out = self.maxpool(out)  # (bs, 1, embedding, T // m)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv2(out)
        out = out.squeeze(1)  # (bs, embedding, T // m)
        out = out.permute(0, 2, 1)  # (bs, T // m, embedding)
        patches = self.SiLU(out)

        patches += self.SinPositionalEncoding(patches)
        out = self.contex_encoder(patches)
        return self.predict_head(torch.mean(out, dim=1))


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
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        patch_size = config['patch_size']
        k = 40
        m = 5
        # Embedding Layer -----------------------------------------------------------
        self.Norm = nn.BatchNorm1d(channel_size)
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(channel_size, 1))

        self.spatial_padding = nn.ReflectionPad2d((int(np.floor((k - 1) / 2)), int(np.ceil((k - 1) / 2)), 0, 0))
        self.spatialwise_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))
        self.spatialwise_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))

        self.SiLU = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, m), stride=(1, m))



        '''
        self.embed_layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=[1, 3], padding='same'),
                                          nn.BatchNorm2d(16),
                                          nn.GELU())

        self.embed_layer1_2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=[channel_size, 1], padding='valid'),
                                            nn.BatchNorm2d(16),
                                            nn.GELU())

        
        '''
        # self.input_projection = nn.Conv1d(channel_size, emb_size, kernel_size=patch_size, stride=patch_size)
        self.SinPositionalEncoding = PositionalEmbedding(int(seq_len/m), emb_size)
        self.contex_encoder = Encoder(config)
        self.predict_head = nn.Linear(emb_size, config['num_labels'])


    def forward(self, x):
        # x = self.embed_layer1(x.unsqueeze(1))
        # x = self.embed_layer1_2(x)
        # patches = self.input_projection(x.squeeze(2)).transpose(1, 2)
        # input is (bs, 1, C, T)
        # x = self.Norm(x)
        # x = gaussrank_transform(x)
        out = x.unsqueeze(1)
        out = self.depthwise_conv(out)  # (bs, embedding, 1 , T)
        out = out.permute(0, 2, 1, 3)  # (bs, 1, embedding, T)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv1(out)  # (bs, 1, embedding, T)
        out = self.SiLU(out)
        out = self.maxpool(out)  # (bs, 1, embedding, T // m)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv2(out)
        out = out.squeeze(1)  # (bs, embedding, T // m)
        out = out.permute(0, 2, 1)  # (bs, T // m, embedding)
        patches = self.SiLU(out)
        patches += self.SinPositionalEncoding(patches)
        out = self.contex_encoder(patches)
        # out = out.view(out.shape[0], -1)
        return self.predict_head(torch.mean(out, dim=1))


class ConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = 'None'
        self.Rel_pos_encode = 'eRPE'
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(emb_size),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        return out


class Conv(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']

        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size, kernel_size=[1, 3], padding='same'),
                                         nn.BatchNorm2d(emb_size),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())
        self.Max = nn.MaxPool2d(kernel_size=[1, 3], stride=[1, 3])
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.C_out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embed_layer(x)
        x = self.embed_layer2(x)
        x = self.Max(x)
        x = x.squeeze(2)
        x = self.gap(x)
        C_out = self.C_out(x.squeeze(-1))
        return C_out


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


class EEG_FeatureExtractor(nn.Module):
    # based on "A deep learning architecture for temporal sleep stage
    # classification using multivariate and multimodal time series"
    def __init__(self, config, num_classes):
        """
        C: number of EEG channels
        T: number of timepoints in a window
        k: length of spatial filters (i.e. how much you look in time)
        m: maxpool size
        n_spatial_filters: number of spatial filters
        embedding_dim: embedding dimension (D)
        """
        # input is (1, C, T) <-- notation (channels, dim1, dim2) is different than paper (dim1, dim2, channels)
        super().__init__()
        C, T = config['Data_shape'][1], config['Data_shape'][2]
        embedding_dim = config['emb_size']
        k = 40
        m = 7
        dropout_prob = 0.5
        n_spatial_filters = 1

        self.Norm = nn.BatchNorm1d(C)
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=C, kernel_size=(C, 1))
        self.spatial_padding = nn.ReflectionPad2d((int(np.floor((k - 1) / 2)), int(np.ceil((k - 1) / 2)), 0, 0))
        self.spatialwise_conv1 = nn.Conv2d(in_channels=1, out_channels=n_spatial_filters, kernel_size=(1, k))
        self.spatialwise_conv2 = nn.Conv2d(in_channels=n_spatial_filters, out_channels=n_spatial_filters,
                                           kernel_size=(1, k))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, m), stride=(1, m))
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
        self.linear = nn.Linear(n_spatial_filters * C * ((T // m) // m), num_classes)
        # self.out = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # input is (bs, 1, C, T)
        bs = x.shape[0]
        x = self.Norm(x)
        # x = gaussrank_transform(x)
        out = x.unsqueeze(1)
        out = self.depthwise_conv(out).transpose(1, 2)  # (bs, C, 1, T)
        # out = out.permute(0, 2, 1, 3)  # (bs, 1, C, T)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv1(out)  # (bs, n_spatial_filters, C, T)
        out = self.relu(out)
        out = self.maxpool(out)  # (bs, n_spatial_filters, C, T // m)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv2(out)  # (bs, n_spatial_filters, C, T // m)
        out = self.relu(out)
        out = self.maxpool(out)  # (bs, n_spatial_filters, C, (T // m) // m)
        out = out.view(bs, -1)  # (bs, n_spatial_filters * C * ((T // m) // m))
        out = self.dropout(out)
        out = self.linear(out)  # (bs, embedding_dim)
        # out = self.out(out)
        return out


def gaussrank_transform(tensor):
    # Flatten the tensor to rank values across all variables and time steps
    flat_tensor = tensor.view(tensor.size(0), -1)
    # Sort the tensor and get the sorted indices
    sorted_tensor, indices = torch.sort(flat_tensor, dim=1)
    # Compute the rank values
    ranks = torch.argsort(indices, dim=1)
    # Add 1 to ranks to start from 1 instead of 0
    ranks = ranks + 1
    # Compute the probability values based on the rank and tensor size
    probs = (ranks - 0.375) / (flat_tensor.size(1) + 0.25)
    # Apply the inverse cumulative distribution function of the standard normal distribution
    gaussranked = torch.erfinv(2 * probs - 1) * (2**0.5)
    # Reshape the tensor to its original shape
    gaussranked = gaussranked.view(tensor.size())

    return gaussranked


