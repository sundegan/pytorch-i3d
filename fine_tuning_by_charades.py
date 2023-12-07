"""使用charades数据集对在ImageNet和Kinetics数据集上预训练的模型进行微调的代码实现。"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

import videotransforms
from charades_dataset import Charades as Dataset
from pytorch_i3d_model import InceptionI3d

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')  # 表示模型处理的是RGB数据还是光流数据
parser.add_argument('-save_model', type=str)  # 模型保存路径
parser.add_argument('-root', type=str)  # 数据集的根目录
args = parser.parse_args()


def run(init_lr=0.1, max_steps=64e3, mode='rgb', root='data/charades/Charades_v1_rgb',
        train_split='data/charades/charades.json', batch_size=8 * 5, save_model=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置数据预处理，例如随机裁剪、水平翻转等
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # 加载Charades数据集，创建训练集和测试集的DataLoader
    dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    # 模型设置，根据不同的模式(RGB或光流)初始化I3D模型，加载预训练的权重参数，并替换最后的分类层以适应当前的数据集
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(157)
    i3d.to(device)
    i3d = nn.DataParallel(i3d)

    # 定义学习率、优化器和学习率调度器
    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    num_steps_per_update = 4  # 累加梯度
    steps = 0
    # 开始训练
    while steps < max_steps:  # for epoch in range(num_epochs):
        print(f'Step {steps}/{max_steps}')
        print('-' * 10)

        # 每个周期都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train()
            else:
                i3d.eval()

            tot_loss = 0.0  # 一个周期中所有迭代的总损失，是定位损失(loc_loss)和分类损失(cls_loss)的加权和
            tot_loc_loss = 0.0  # 总定位损失
            tot_cls_loss = 0.0  # 总分类损失
            num_iter = 0  # 迭代次数
            optimizer.zero_grad()

            for data in dataloaders[phase]:
                num_iter += 1
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                per_frame_logits, _ = i3d(inputs)  # 计算每帧的logit

                # 使用上采样调整per_frame_logits的大小和输入视频帧的数量相同
                # 将模型输出的logits调整到与原始输入视频中的帧数相匹配。这是必要的，因为在视频处理中，由于各种原因
                # 例如时间池化或卷积层的步长设置，模型在时间维度上的输出可能与输入的帧数不同。
                # 通过上采样，可以确保模型输出的每一帧都有一个对应的预测结果，这对于后续的时间定位和分类任务非常重要。
                # pytorch-i3d原代码使用的是线性插值linear方法，我这里改成了双线性插值bilinear方法。
                t = inputs.size(2)  # 获取视频帧数
                per_frame_logits = F.interpolate(per_frame_logits, size=t, mode='bilinear')

                # 计算定位损失
                # per_frame_logits和labels的形状为[B, C, T]
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data[0]

                # 计算分类损失 (在时间维度进行最大池化)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                              torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data[0]

                # 实际损失由定位损失和分类损失加权和得到
                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data[0]
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        print(
                            f'{phase} Loc Loss: {tot_loc_loss / (10 * num_steps_per_update):.4f}, Cls Loss: {tot_cls_loss / (10 * num_steps_per_update):.4f}, Tot Loss: {tot_loss / 10:.4f}')
                        # 保存模型
                        torch.save(i3d.module.state_dict(), save_model + str(steps).zfill(6) + '.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                print(
                    f'{phase} Loc Loss: {tot_loc_loss / num_iter:.4f}, Cls Loss: {tot_cls_loss / num_iter:.4f}, Tot Loss: {(tot_loss * num_steps_per_update) / num_iter:.4f}')


if __name__ == '__main__':
    run(mode=args.mode, root=args.root, save_model=args.save_model)
