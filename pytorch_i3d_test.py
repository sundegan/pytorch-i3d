"""检查I3D模型在各种情况下是否可以正确构建并生成正确的形状。"""

import torch
import unittest

import pytorch_i3d_model as i3d

_IMAGE_SIZE = 224
_NUM_CLASSES = 400


class I3DTest(unittest.TestCase):
    """测试Inception I3D模型，不使用真实数据。"""

    def test_inception_module(self):
        """测试InceptionModule模块和预期是否相符。"""
        inception = i3d.InceptionModule(3, [1, 2, 3, 4, 5, 6], name='test')
        inputs = torch.rand(5, 3, 64, _IMAGE_SIZE, _IMAGE_SIZE)  # batch=5, channels=3, t=64, h=_IMAGE_SIZE, w=_IMAGE_SIZE
        outputs = inception(inputs)
        self.assertEqual(outputs.shape, (5, 15, 64, _IMAGE_SIZE, _IMAGE_SIZE))  # 1+3+5+6=15

    def test_model_shapes_with_squeeze(self):
        """测试开启`spatial_squeeze` 时输出的形状是否正确。"""
        i3d_model = i3d.InceptionI3d(num_classes=_NUM_CLASSES, final_endpoint='Predictions')
        inputs = torch.randn(5, 3, 64, _IMAGE_SIZE, _IMAGE_SIZE)
        predictions, _ = i3d_model(inputs)
        self.assertEqual(predictions.shape, (5, _NUM_CLASSES))

    def test_model_shapes_without_squeeze(self):
        """测试关闭`spatial_squeeze` 时输出形状的变化。"""
        i3d_model = i3d.InceptionI3d(num_classes=_NUM_CLASSES, spatial_squeeze=False, final_endpoint='Predictions')
        inputs = torch.randn(5, 3, 64, _IMAGE_SIZE, _IMAGE_SIZE)
        predictions, _ = i3d_model(inputs)
        self.assertEqual(predictions.shape, (5, _NUM_CLASSES, 1, 1))

    def test_model_shape_with_logits_and_squeeze(self):
        """测试当`final_endpoint`为`Logits`时的形状是否正确。"""
        i3d_model = i3d.InceptionI3d(num_classes=_NUM_CLASSES, final_endpoint='Logits')
        inputs = torch.randn(5, 3, 64, _IMAGE_SIZE, _IMAGE_SIZE)
        logits, _ = i3d_model(inputs)
        self.assertEqual(logits.shape, (5, _NUM_CLASSES, 7))

    def test_model_shape_with_logits_and_without_squeeze(self):
        """测试当`final_endpoint`为`Logits`时且关闭`spatial_squeeze`时输出的形状是否正确。"""
        i3d_model = i3d.InceptionI3d(num_classes=_NUM_CLASSES, spatial_squeeze=False, final_endpoint='Logits')
        inputs = torch.randn(5, 3, 64, _IMAGE_SIZE, _IMAGE_SIZE)
        logits, _ = i3d_model(inputs)
        self.assertEqual(logits.shape, (5, _NUM_CLASSES, 7, 1, 1))

    def test_init_errors(self):
        # 测试无效的 `final_endpoint` 字符串。
        with self.assertRaises(ValueError):
            _ = i3d.InceptionI3d(num_classes=_NUM_CLASSES, final_endpoint='Conv3d_1a_8x8')

        # 输入的高度和宽度尺寸应为 _IMAGE_SIZE。
        with self.assertRaises(ValueError):
            i3d_model = i3d.InceptionI3d(num_classes=_NUM_CLASSES)
            inputs = torch.randn(1, 3, 16, 192, 192)
            _ = i3d_model(inputs)


if __name__ == '__main__':
    unittest.main()
