import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class MaxPool3dSamePadding(nn.MaxPool3d):
    """
    MaxPool3dSamePadding类是对PyTorch中的MaxPool3d的扩展，添加了"same"的填充功能。
    在进行最大池化操作时，会自动计算并应用必要的填充，以确保输出特征图的尺寸和输入特征图的尺寸在空间维度上尽可能相似。
    """

    def compute_pad(self, dim, s):
        """
        compute_pad()用于计算在特定维度上需要的填充量，根据池化核大小、步长和输入特征图的尺寸计算。
        Args:
            dim: 指定计算填充量的维度。在三维情况下，dim可以是0、1、2，分别对应于时间维度(深度)、高度和宽度。
            s: 输入特征图在指定维度上的大小。比如，如果dim为1(高度)，s就是输入特征图的高度。

        Returns: 函数返回计算得到的填充量，表示在指定维度上总共需要添加的填充数量。
        """
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # 获取输入x的尺寸
        (batch, channel, t, h, w) = x.size()
        # 计算输出尺寸
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # 计算各个维度的填充量
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # 将每个维度的填充量均匀分配到前后（或上下、左右）两侧
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        # 使用F.pad()对输入x应用计算的填充量后进行最大池化
        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super().forward(x)


class Unit3D(nn.Module):
    """
    Unit3D是一个基本的3D卷积单元，是构建I3D模型的基础模块，包含基本的3D卷积层、可选的批量归一化层和激活函数层。
    通过组合3D卷积、批量归一化和激活函数，能够提取数据中的时空特征，同时通过动态填充机制，还能够在执行卷积操作时保持特征图的空间尺寸。
    """

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        """
        初始化Unit3D模块。
        Args:
            in_channels: 输入通道数。
            output_channels: 输出通道数。
            kernel_shape: 卷积核形状，是一个三元组，分别代表深度、高度和宽度上的卷积核尺寸。
            stride: 卷积的步长。
            padding: 卷积的填充量，默认值为0表示不填充。
            activation_fn: 激活函数，默认为ReLU函数。
            use_batch_norm: 是否使用批量归一化。
            use_bias: 表示卷积层是否使用偏置项。
            name: 模块的名称。
        """
        super().__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        # 定义3D卷积层
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,  # 这里总是设为0，通过在forward中动态地进行填充而不是静态地设置一个固定的填充值
                                bias=self._use_bias)

        # 定义批量归一化层（可选地）
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        # 进行填充后进行三维卷积
        x = self.conv3d(x)
        # 进行批量归一化(可选地)
        if self._use_batch_norm:
            x = self.bn(x)
        # 如果指定了激活函数，则应用激活函数(默认为ReLU)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    """
    InceptionModule是一个实现Inception架构的基本模块。
    Inception架构最初在GoogleNet(Inception v1)中提出，其核心思想是在同一层内并行使用不同尺寸的卷积核，从而能够在单个层级上捕捉不同尺度的空间特征。
    papers：https://arxiv.org/abs/1409.4842
    """

    def __init__(self, in_channels, out_channels_list, name):
        """
        在该构造函数中，定义了四个分支：Branch_0、Branch_1、Branch_2、Branch_3，
        分别使用不同大小的卷积核同时捕获输入数据在不同尺度上的特征，每个分支输出的特征图数量由out_channels数组指定。
        b0：一个1*1*1的卷积核，主要用于捕捉局部的、细粒度的特征。
        b1a和b1b：先是一个1*1*1的卷积核减少通道数（有助于降低计算量），然后b1b使用3x3x3的卷积核捕获中等尺度的特征。
        b2a和b2b：类似于b1，但可能使用不同数量的通道或不同尺寸的卷积核。b2a用于降维，而b2b用于捕捉更大尺度的特征。
        b3a和b3b：先是一个3*3*3的最大池化操作可以捕获更广泛的特征，随后使用1x1x1的卷积核进行降维。
        Args:
            in_channels: 输入通道数。
            out_channels_list: 输出通道数，是一个数组，分别指定多个分支的卷积操作的输出通道数。
            name: 模块名称。
        """
        super().__init__()

        # 分支0
        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels_list[0], kernel_shape=(1, 1, 1),
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        # 分支1
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels_list[1], kernel_shape=(1, 1, 1),
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels_list[1], output_channels=out_channels_list[2],
                          kernel_shape=(3, 3, 3),
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        # 分支2
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels_list[3], kernel_shape=(1, 1, 1),
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels_list[3], output_channels=out_channels_list[4],
                          kernel_shape=(3, 3, 3),
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        # 分支3
        self.b3a = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels_list[5], kernel_shape=(1, 1, 1),
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        # 输入数据x依次通过四个分支，然后将四个分支的输出在通道维度(dim=1)上拼接起来，
        # 拼接后的总输出通道数=out_channels[0]+out_channels[2]+out_channels[4]+out_channels[5]。
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """基于Inception-v1的I3D模型。
    I3D模型结构在《Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset》这篇论文首次提出,
     https://arxiv.org/pdf/1705.07750v1.pdf。
    同时也参考了Inception结构，关于Inception模型结构，在论文《Going deeper with convolutions》进行了介绍，
     https://arxiv.org/pdf/1409.4842v1.pdf。
    """

    # 模型的所有有效端点按顺序排列。在构造过程中，`final_endpoint`之前的所有端点都作为第二个返回值在字典中返回。
    # 模型端点的作用是可以选择性构建，根据需要选择模型构建的深度，模型只会构建到指定的`final_endpoint`这一层。
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'AVGLogits',  # 新增的end_point，用来对time维度进行平均
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """初始化I3D模型实例。
        Args:
          num_classes: logit层的输出数量（默认为400，和Kinetics-400数据集的类别数相匹配）
          spatial_squeeze: 是否在返回之前压缩logits的空间维度(默认为True)。
          final_endpoint: 模型有多个可能的端点。`final_endpoint`指定了要构建模型的最后一个端点。
                            除了`final_endpoint`端点会被返回，在`final_endpoint`之前的所有端点也都会以字典的形式返回。
                            `final_endpoint`必须是InceptionI3d.VALID_ENDPOINTS中列出的有效值中的一个（默认为'Logits'）。
          name: 模块名称（可选地）。
        Raises:
            如果`final_endpoint`不能被识别，则会报错。
        """

        # 验证是否为有效端点，初始化实例变量和父类
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super().__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        # 定义并构建网络层结构
        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=(7, 7, 7),
                                            stride=(2, 2, 2), name=name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=(1, 1, 1),
                                            name=name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=(3, 3, 3),
                                            name=name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(64 + 128 + 32 + 32, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2))
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=(1, 1, 1),
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()

    # 更改logit层的输出类别数。
    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=(1, 1, 1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    # 构建网络，将定义的端点添加到模块中。
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    # 定义前向传播过程
    def forward(self, x):
        """
        Args:
            x: 输入数据必须是[batch_size, num_channels, num_frames, 224, 224]形状的。
        """

        # 检查输入数据的尺寸是否符合要求，num_frames最少为9，否则会在卷积过程中因为数据尺寸小于卷积核大小而报错
        if x.size(2) <= 8 or x.size(3) < 224 or x.size(4) < 224:
            raise ValueError("输入尺寸需要至少为 [batch, channels, 9, 224, 224]")

        # 保存每个端点的输出
        end_points = {}
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # 获取对应端点的模块，并对输入x应用该模块。
                end_points[end_point] = x  # 保存每一层的输出
                if self._final_endpoint == end_point:  # 返回当前层的输出和所有收集到的端点输出
                    return x, end_points

        end_point = 'Logits'
        # self.avg_pool(x)：应用全局平均池化。这通常用于减少特征图的维度，为全连接层做准备。
        # self.dropout(): 应用dropout操作，这有助于防止过拟合。
        # self.logits(): 将处理后的数据传递到logits层。这是一个全连接层，通常用于生成最终的分类得分。
        x = self.logits(self.dropout(self.avg_pool(x)))

        # 如果进行空间压缩，将移除末尾两个维度为1的维度，使输出的形状更加紧凑。
        #   比如x的形状是 [batch_size, classes, time, 1, 1]，
        #   经过第一个squeeze(3)之后，x的形状会变为[batch_size,classes, time, 1]，
        #   第二个squeeze(3)将再次尝试移除现在的第4维other，最终输出形状为[batch_size, num_classes, time]。
        # 如果不应用空间压缩，最后两个维度将保留为单元素维度，形状将是[batch_size, num_classes, time, 1, 1]。
        logits = x.squeeze(3).squeeze(3) if self._spatial_squeeze else x
        end_points[end_point] = logits
        if self._final_endpoint == end_point:
            return logits, end_points

        # 针对num_frames维度进行平均，最终输出形状变为[batch_size, classes]
        end_point = 'AVGLogits'
        averaged_logits = torch.mean(logits, dim=2)
        end_points[end_point] = averaged_logits
        if self._final_endpoint == end_point:
            return averaged_logits, end_points

        # 对classes维度进行softmax预测概率
        end_point = 'Predictions'
        predictions = torch.softmax(averaged_logits, dim=1)
        end_points[end_point] = predictions
        return predictions, end_points

    # 从输入数据x中提取特征。这个方法用于获取模型中间层的输出，而不仅仅是最终的分类层（logits层）的输出。
    # 这可以将特征用于多种下游任务，比如特征可视化、进一步的特征处理或在其他任务中使用这些特征。
    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)