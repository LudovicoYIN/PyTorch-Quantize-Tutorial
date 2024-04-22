# Pytorch 量化

## Pytorch支持量化类型
1. Post Training Dynamic Quantization，模型训练完毕后的动态量化
2. Post Training Static Quantization，模型训练完毕后的静态量化；
3. QAT（Quantization Aware Training），模型训练中开启量化。
---
## Tensor量化
1. 具体demo见 tensor_quantize_demo.py
---
## Post Training Dynamic Quantization
1. 对训练后的模型权重执行动态量化，将浮点模型转换为动态量化模型，仅对模型权重进行量化，偏置不会量化。
2. 默认情况下，仅对 Linear 和 RNN 变体量化 (因为这些layer的参数量很大，收益更高)。
---
`torch.quantization.quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False)`
1. 参数:
   1. model：浮点模型 
   2. qconfig_spec： 
      1. 集合：比如： qconfig_spec={nn.LSTM, nn.Linear} 。罗列要量化的
      2. 字典： qconfig_spec = {nn.Linear : default_dynamic_qconfig, nn.LSTM : default_dynamic_qconfig} 
   3. dtype： float16 或 qint8
   4. mapping：就地执行模型转换，原始模块发生变异
   5. inplace：将子模块的类型映射到需要替换子模块的相应动态量化版本的类型