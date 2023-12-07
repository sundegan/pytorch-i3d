# 1. I3D模型Pytorch版本实现（Pytorch version of the I3D Model）
本仓库代码根据[原论文开源代码(TF版本)](https://github.com/google-deepmind/kinetics-i3d)和
[pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)综合修改而来，主要在`pytorch-i3d`的基础上进行修改，是I3D模型的Pytorch版本，
并添加详细的中文注释，易于理解和学习。

# 2. 主要修改
- 增加详细的中文注释和使用说明。 
- 模型代码直接支持Python3和Pytorch2.0版本及以上。
- 对代码风格和结构进行了部分修改，使代码尽可能简洁以及和原论文的开源代码保持相近。
- 修改模型最后的输出部分，增加`AVGLogits`层用来对结果进行平均和降维(原论文在`Logits`层就进行了平均，
但是`pytorch-i3d`把平均放到了`train.py`文件中进行，以及缺少`Predictions`部分，给人带来了一定困扰)，
我这里增加一个`AVGLogits`层专门用来对模型输出的`Logits`进行平均，以及增加`Predictions`对输出最后预测概率，和原论文开源代码保持一致。
- `pytorch-i3d`仓库代码中缺少模型测试文件`pytorch_i3d_test.py`和对模型进行评估的脚本`evaluate_sample.py`，这里增加这两个文件。
- `pytorch-i3d`仓库中根据`Charades`数据集进行微调的训练代码名为`train.py`，
我这里将其更名为`fine_tuning_by_charades.py`并增加详细注释和进行小修改，同时完善项目目录结构。

# 3. 各文件说明
```text
pytorch-i3d
├── LICENSE.txt
├── README.md
├── charades_dataset.py         # 负责从给定的文件和目录中读取视频信息和帧，只获取视频中的一部分帧，用于模型微调
├── charades_dataset_full.py    # 负责从给定的文件和目录中读取视频信息和帧，获取视频的全部帧（GPU显存和算力较强可以选择使用全部帧）
├── checkpoints                 # 预训练模型的检查点
│   ├── flow_charades.pt        # 在ImageNet、Kinetics、Charades三个数据集上训练的光流模型检查点
│   ├── flow_imagenet.pt        # 在ImageNet、Kinetics两个数据集上训练的光流模型检查点
│   ├── rgb_charades.pt         # 在ImageNet、Kinetics、Charades三个数据集上训练的RGB模型检查点
│   └── rgb_imagenet.pt         # 在ImageNet、Kinetics两个数据集上训练的RGB模型检查点
├── data                        # 使用到的数据文件
│   ├── charades                # charades数据集及其数据划分的json文件
│   │   ├── Charades_v1_rgb
│   │   │      └── README.md    # 需自己去官网进行数据下载
│   │   └── charades.json       # 数据划分的json文件，根据这个文件夹把数据集划分为训练集、验证集和测试集
│   └── sample                  # evaluate_sample.py中使用到的UCF101样本数据（https://www.crcv.ucf.edu/data/UCF101.php）
│       ├── label_map.txt       # UCF101数据集标签
│       ├── v_CricketShot_g04_c01_flow.gif
│       ├── v_CricketShot_g04_c01_flow.npy
│       ├── v_CricketShot_g04_c01_rgb.gif
│       └── v_CricketShot_g04_c01_rgb.npy
├── evaluate_sample.py          # 对UCF101中的样本数据进行评估
├── extract_features.py         # 使用预训练的I3D模型提取Charades数据集的特征保存为Numpy数组方便后续使用
├── fine_tuning_by_charades.py  # 使用Charades数据集对在ImageNet和Kinetics数据集上预训练的模型进行微调
├── pytorch_i3d_model.py        # Pytorch版本I3D模型定义
├── pytorch_i3d_test.py         # 对I3D模型进行测试
└── videotransforms.py          # 视频预处理
```

# 4. 链接
- I3D论文：https://arxiv.org/abs/1705.07750v3
- Inception论文：https://arxiv.org/abs/1409.4842
- Kinetics-I3D：https://github.com/google-deepmind/kinetics-i3d
- Pytorch-I3D：https://github.com/piergiaj/pytorch-i3d
- Charades数据集：https://prior.allenai.org/projects/charades
- charades.json文件：https://github.com/piergiaj/super-events-cvpr18/blob/master/data/charades.json

