"""使用预训练的模型对样本进行预测。"""

import torch
import torch.nn as nn
import numpy as np

from pytorch_i3d_model import InceptionI3d

_IMAGE_SIZE = 224
_SAMPLE_VIDEO_FRAMES = 79
_NUM_CLASSES = 400

_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}
_CHECKPOINT_PATHS = {
    'rgb': 'models/rgb_imagenet.pt',
    'flow': 'models/flow_imagenet.pt',
}
_LABEL_MAP_PATH = 'data/label_map.txt'


# 加载标签映射
def load_label_map(label_map_path):
    with open(label_map_path, 'r') as f:
        return [line.strip() for line in f]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(_LABEL_MAP_PATH)

    # 设置评估模式，joint表示双流模式，综合rgb模型和flow模型两者的预测；rgb表示只使用单流rgb模型评估rgb图像；flow表示只使用单流flow模型评估flow图像
    eval_type = 'joint'  # or 'rgb', 'flow'
    if eval_type not in ['rgb', 'flow', 'joint']:
        raise ValueError('错误的评估类型，eval_type必须是rgb, flow, joint三者之一')

    # 根据不同的评估模式加载不同的模型权重
    if eval_type in ['rgb', 'joint']:
        rgb_model = InceptionI3d(_NUM_CLASSES, in_channels=3, final_endpoint='AVGLogits')
        rgb_model.load_state_dict(torch.load(_CHECKPOINT_PATHS['rgb']))
        rgb_model.to(device)
        rgb_model.eval()

    if eval_type in ['flow', 'joint']:
        flow_model = InceptionI3d(_NUM_CLASSES, in_channels=2, final_endpoint='AVGLogits')
        flow_model.load_state_dict(torch.load(_CHECKPOINT_PATHS['flow']))
        flow_model.to(device)
        flow_model.eval()

    # 加载并且处理样本数据
    if eval_type in ['rgb', 'joint']:
        rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
        # # 将numpy数组格式[batch, frames, h, w, c]转换为pytorch的数据格式[batch, c, frames, h, w]
        rgb_sample = rgb_sample.transpose((0, 4, 1, 2, 3))
        rgb_sample = torch.from_numpy(rgb_sample).to(device)

    if eval_type in ['flow', 'joint']:
        flow_sample = np.load(_SAMPLE_PATHS['flow'])
        # 将numpy数组格式[batch, frames, h, w, c]转换为pytorch的数据格式[batch, c, frames, h, w]
        flow_sample = flow_sample.transpose((0, 4, 1, 2, 3))
        flow_sample = torch.from_numpy(flow_sample).to(device)

    # 模型预测
    with torch.no_grad():
        if eval_type in ['rgb', 'joint']:
            rgb_logits, _ = rgb_model(rgb_sample)
            model_logits = rgb_logits
        if eval_type in ['flow', 'joint']:
            flow_logits, _ = flow_model(flow_sample)
            model_logits = flow_logits
        if eval_type == 'joint':
            model_logits = rgb_logits + flow_logits
        model_predictions = nn.functional.softmax(model_logits, dim=1)

    # 打印输出
    # 最终结果和https://github.com/google-deepmind/kinetics-i3d中的样本输出结果基本一样
    out_logits = np.array(model_logits[0])
    out_predictions = np.array(model_predictions[0])
    sorted_indices = np.argsort(out_predictions)[::-1]  # 从大到下对out_predictions排序返回对应的下标索引
    print(f'Norm of logits: {np.linalg.norm(out_logits):.6f}')  # 打印logits的范数
    print('\nTop classes and probabilities')
    for index in sorted_indices[:20]:   # 得到每个样本的前20个预测值
        print("{:<15.6g} {:<15.4f} {}".format(out_predictions[index], out_logits[index], label_map[index]))


if __name__ == '__main__':
    main()
