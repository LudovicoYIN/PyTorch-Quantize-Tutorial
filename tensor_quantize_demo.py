import torch

"""
需要量化时，则需要能够表示量化数据的Tensor
pytorch1.1之后引入了Quantized Tensor
可以表示int8/uint8/int32类型
有scale, zero_point的属性
"""
x = torch.rand(2, 3, dtype=torch.float32)

x_quantize = torch.quantize_per_tensor(x, scale=0.5, zero_point=8, dtype=torch.quint8)
print(x)  # float32 Tensor
print(x_quantize)  # Quantized Tensor
print(x_quantize.int_repr())  # Quantized Tensor to uint8
print()
"""
float32 Tensor与量化后数据的关系为：
x_quant = round(x / scale + zero_point)
当x为0时：
xq = round(zero_point) = zero_point
通过scale和zero_point进行反量化：
x_dequantize = (xq - zero_point) * scale
"""
x_dequantize = x_quantize.dequantize()
print(x_dequantize)
print()
"""
x_dequantize 和 x 可以看到精度有损失
在PyTorch中，选择合适的scale和zp的工作就由各种observer来完成
Tensor的量化支持两种模式：per tensor 和 per channel
Per tensor 是说一个tensor里的所有value按照同一种方式去scale和offset；
Per channel是对于tensor的某一个维度（通常是channel的维度）上的值按照一种方式去scale和offset也就是一个tensor里有多种不同的scale和offset的方式（组成一个vector）
"""