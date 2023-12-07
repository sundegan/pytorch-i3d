"""使用I3D模型来处理视频数据，并将提取的特征保存为NumPy数组，这些特征可以用于后续的机器学习或深度学习任务。"""

import torch
from torchvision import transforms
import videotransforms
import numpy as np
from pytorch_i3d_model import InceptionI3d
from charades_dataset_full import Charades as Dataset

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')  # 模式
parser.add_argument('-load_model', type=str)  # 模型路径
parser.add_argument('-root', type=str)  # 数据根目录
parser.add_argument('-gpu', type=str)  # 使用的GPU
parser.add_argument('-save_dir', type=str)  # 保存目录
args = parser.parse_args()


def run(mode='rgb', root='data/charades/Charades_v1_rgb', split='data/charades/charades.json',
        batch_size=1, load_model='checkpoints/rgb_imagenet.pt', save_dir='out'):
    """
    特征提取的主要函数。
    Args:
        mode: 模型模式，rgb或者flow。
        root: 数据集根目录。
        split: 包含视频数据集分割信息的JSON文件的路径。
        batch_size: 批次大小。
        load_model: 需要加载的模型检查点路径。
        save_dir: 数据保存路径。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置训练集和验证集
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    # 模型设置
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)  # 替换为Charades的类别数量
    i3d.load_state_dict(torch.load(load_model))
    i3d.to(device)

    for phase in ['train', 'val']:
        i3d.eval()
                    
        # 迭代数据
        for data in dataloaders[phase]:
            inputs, labels, name = data
            if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
                continue

            b, c, t, h, w = inputs.shape
            if t > 1600:
                features = []
                for start in range(1, t-56, 1600):
                    end = min(t-1, start+1600+56)
                    start = max(1, start-48)
                    inputs = torch.from_numpy(inputs.numpy()[:,:,start:end]).to(device)
                    features.append(i3d.extract_features(inputs).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
                np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
            else:
                inputs = inputs.to(device)
                features = i3d.extract_features(inputs)
                np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())


if __name__ == '__main__':
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
