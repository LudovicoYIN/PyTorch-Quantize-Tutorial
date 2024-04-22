"""
只量化权重，不量化激活
"""
import torch
from torch import nn


class DemoModel(torch.nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model_fp32 = DemoModel()
    # 创建一个量化的模型实例
    model_int8 = torch.quantization.quantize_dynamic(
        model=model_fp32,  # 原始模型
        qconfig_spec={torch.nn.Linear},  # 要动态量化的NN算子
        dtype=torch.qint8)  # 将权重量化为：float16 \ qint8

    print(model_fp32)
    print(model_int8)

    # 运行模型
    input_fp32 = torch.randn(1, 1, 2, 2)
    output_fp32 = model_fp32(input_fp32)
    print(output_fp32)

    output_int8 = model_int8(input_fp32)
    print(output_int8)
