import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

import cv2


def video_to_tensor(pic):
    """
    将numpy数组(T x H x W x C)转化为张量(C x T x H x W)。
    Args:
         pic (numpy.ndarray): 需要转换为张量形式的视频.
    Returns:
         Tensor: 转换后的视频.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, start, num):
    """
    从指定视频中加载指定数量的RGB帧，从给定帧开始。
    它读取每一帧作为图像，必要时调整大小，规范化像素值，并将它们存储在数组中。
    Args:
        image_dir: 存储视频帧图像的目录。
        vid: 当前处理的视频的标识符或名称。
        start: 开始加载帧的起始索引。
        num: 要加载的帧的数量。

    """
    frames = []  # 初始化一个空列表frames，用于存储加载的帧。
    for i in range(start, start + num):
        # 使用cv2.imread加载图像。构造的文件路径是image_dir/vid/vid-i.jpg，其中i是格式化为六位数的帧编号（使用zfill(6)）。
        # [:, :, [2, 1, 0]] 将图像从 BGR（OpenCV 默认格式）转换为 RGB 格式
        img = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + '.jpg'))[:, :, [2, 1, 0]]
        # 检查图像的宽度或高度是否小于226像素。如果是，就需要对图像进行缩放，确保图像尺寸最小为226。
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)  # 计算缩放后的目标尺寸与当前尺寸的差值。
            sc = 1 + d / min(w, h)  # 计算缩放因子。
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)  # 使用cv2.resize通过计算出的缩放因子sc对图像进行缩放。
        img = (img / 255.) * 2 - 1  # 将图像像素值从[0, 255]范围标准化到[-1, 1]。这是一种常见的图像预处理步骤，有助于改进模型的性能。
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)  # 将帧列表转换为numpy数组，指定数据类型为 float32


def load_flow_frames(image_dir, vid, start, num):
    """
    用于加载和处理光流帧。
    """
    frames = []
    for i in range(start, start + num):
        # 分别加载光流的x分量和y分量的图像。使用cv2.IMREAD_GRAYSCALE参数，因为光流的每个分量通常表示为灰度图像（单通道）。
        imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

        # 对图像进行缩放
        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)
        # 将光流图像的像素值从 [0, 255] 范围标准化到 [-1, 1]。
        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        # imgx和imgy是两个单通道（灰度）图像，分别表示光流的水平（x）和垂直（y）分量。
        # np.asarray([imgx, imgy])将x分量和y分量的图像合并为一个数组，数组的形状为(2, h, w)，其中2是通道数（两个光流分量）。
        # .transpose([1, 2, 0])将数组的形状从 (2, h, w) 变为 (h, w, 2)
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=157):
    """
    make_dataset函数负责从给定的文件和目录中读取视频信息和帧，创建包含视频ID、标签、持续时间和帧数的数据集。
    Args:
        split_file: 包含视频数据集分割信息的JSON文件的路径。这个文件定义了哪些视频属于训练集、验证集或测试集。
        split: 指定要创建的数据集类型（例如 'training', 'validation', 'test'）。
        root: 视频帧或光流帧的根目录。
        mode: 指定数据集的模式，比如 'rgb' 或 'flow'，这决定了加载的帧类型。
        num_classes: 数据集中动作类别的总数，默认值为157。

    Returns: 返回构建好的数据集列表，其中每个元素包含视频 ID、标签数组、视频持续时间和帧数。

    Examples: split_file的格式如下：
        {
            "video1": {
                "subset": "training",
                "duration": 120.0,
                "actions": [
                    [0, 30.5, 45.0],
                    [2, 60.0, 75.5]
                ]
            },
            // ... 更多视频
        }
        video1 表示视频的标识，对于charades是一个由六个字符组成的标识ID。
        subset 表示该视频所属的数据集。
        duration 表示视频的持续时间（秒）。
        actions 是一个列表，包含了视频中发生的动作。每个动作是一个列表，包含三个元素：[动作类别, 动作开始时间, 动作结束时间]。
        例如，[0, 30.5, 45.0] 表示类别为 0 的动作从视频的第 30.5 秒开始，到第 45.0 秒结束。
    """
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0  # 初始化计数器。
    # 遍历数据中的每个视频ID。
    for vid in data.keys():
        # 如果视频的子集不是指定的split（如 'training'），则跳过该视频。
        if data[vid]['subset'] != split:
            continue

        # 检查视频的帧是否真的存在于给定的root路径下。
        if not os.path.exists(os.path.join(root, vid)):
            continue

        # 计算视频中帧的数量。
        num_frames = len(os.listdir(os.path.join(root, vid)))

        # 如果模式是 'flow'（光流），将帧数除以 2，因为光流数据通常包括水平（x）和垂直（y）两个分量。
        if mode == 'flow':
            num_frames = num_frames // 2

        # 如果帧数少于 66，跳过这个视频。这通常是为了确保视频长度足以进行某些类型的处理或分析。
        if num_frames < 66:
            continue

        # 初始化一个标签数组，形状为 (类别数, 帧数)，用于存储每个帧的类别标签。
        label = np.zeros((num_classes, num_frames), np.float32)

        # 计算视频的帧率（每秒帧数）。
        fps = num_frames / data[vid]['duration']

        # 遍历视频的每个动作标注ann。对于每个帧，检查该帧是否在动作的时间范围内。如果是，将对应的类别在标签数组中标记为1，实现二元分类。
        # fr/fps计算当前帧的时间戳（秒）。
        for ann in data[vid]['actions']:
            for fr in range(0, num_frames, 1):
                if ann[1] < fr / fps < ann[2]:  # 检查当前帧的时间戳是否在动作发生的时间区间内。ann[1]是动作开始的时间，ann[2]是动作结束的时间。
                    label[ann[0], fr] = 1  # binary classification

        # 将视频ID、标签、视频持续时间和帧数作为一个元组添加到数据集列表中。
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1  # 增加计数器。

    return dataset


class Charades(torch.utils.data.Dataset):
    """
    自定义的PyTorch数据集类，用于处理和加载Charades数据集。
    """
    def __init__(self, split_file, split, root, mode, transforms=None):

        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # 从self.data（在初始化时创建的数据集）中获取相应的视频信息和标签。
        vid, label, duration, num_frames = self.data[index]
        # 用于随机选择一个起始帧start_f来加载一系列连续的视频帧。
        # 这是在处理视频数据，特别是在训练机器学习模型时常用的技术，目的是为了增加数据的多样性和泛化能力。
        # random.randint(1, num_frames - 65)生成一个随机整数，范围从1到num_frames - 65。
        # 为什么使用 num_frames - 65 作为随机数的上界？
        # 在 __getitem__ 方法中，目的是加载一系列连续的64帧。为了确保这64帧是连续的且完全包含在视频内，我们需要从视频的某个点开始选择这64帧。
        # 如果我们选择了靠近视频末尾的起始点，可能会没有足够的帧来形成这个64帧的序列。例如，如果视频总共有100帧，从第95帧开始就无法获取完整的64帧序列。
        # 因此，通过从 num_frames - 65 中选择起始帧，我们可以确保总是有至少64帧可用来加载。这样做有效地防止了选择一个起始点，它太接近视频的末尾，以至于不能提供足够的连续帧。
        start_f = random.randint(1, num_frames - 65)

        # 根据数据集的模式（"rgb" 或 "flow"），它调用 load_rgb_frames 或 load_flow_frames 函数来加载视频帧。
        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_f, 64)
        else:
            imgs = load_flow_frames(self.root, vid, start_f, 64)
        label = label[:, start_f:start_f + 64]

        imgs = self.transforms(imgs)

        # 最后，将加载的帧和标签转换为张量，并返回它们。
        return video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
