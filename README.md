# 1. I3D模型Pytorch版本实现（Pytorch version of the I3D Model）
本仓库代码根据[原论文开源代码(TF版本)](https://github.com/google-deepmind/kinetics-i3d)和
[pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)综合修改而来， 
主要在pytorch-i3d的基础上进行修改，添加了较为详细的中文注释和说明，以及对代码风格进行了部分修改，使代码尽可能和原论文的开源代码保持相近。

# 2. 主要修改
- 增加详细的中文注释。 
- 修改模型最后的输出部分，增加`AVGLogits`层用来对结果进行平均和降维(原论文在`Logits`层就进行了平均，
但是`pytorch-i3d`把平均放到了train.py文件中进行，以及缺少`Predictions`部分，给人带来了一定困扰)，
我这里增加一个`AVGLogits`层专门用来对模型输出的`Logits`进行平均，以及增加`Predictions`对输出最后预测概率，和原论文开源代码保持一致。
- `pytorch-i3d`仓库代码中缺少模型测试文件`pytorch_i3d_test.py`和对模型进行评估的脚本`evaluate_sample.py`，这里增加这两个文件。

# 3. 各文件介绍
