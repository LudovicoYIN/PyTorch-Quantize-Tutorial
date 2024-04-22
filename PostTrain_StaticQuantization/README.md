# Post Training Dynamic Quantization
## quantize_dynamic APi
`torch.quantization.quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False)`
1. quantize_dynamic这个API把一个float model转换为dynamic quantized model
2. model中只有权重被量化，dtype参数可以取值 float16 或者 qint8
3. 当对整个模型进行转换时，默认只对以下的op进行转换：
   1. Linear 
   2. LSTM 
   3. LSTMCell 
   4. RNNCell 
   5. GRUCell
4. 思考：为什么只有这些？
   1. 因为dynamic quantization只是把权重参数进行量化，而这些layer一般参数数量很大，在整个模型中参数量占比极高，因此边际效益高。
   2. 对其它layer进行dynamic quantization几乎没有实际的意义。

---
## qconfig_spec
1. qconfig_spec指定了一组qconfig，具体就是哪个op对应哪个qconfig 
   1. 每个qconfig是QConfig类的实例，封装了两个observer
      1. 这两个observer分别是activation的observer和weight的observer
2. quantize_dynamic使用的是QConfig子类QConfigDynamic的实例，该实例实际上只封装了weight的observer
3. activate就是post process，就是op forward之后的后处理，但在动态量化中不包含
4. observer用来根据四元组（min_val，max_val，qmin, qmax）来计算2个量化的参数：scale和zero_point
5. qmin、qmax是算法提前确定好的，min_val和max_val是从输入数据中观察到的，所以起名叫observer
6. qconfig_spec为None的时候就是默认行为，如果想要改变默认行为，则可以： 
   1. qconfig_spec赋值为一个set，比如：{nn.LSTM, nn.Linear}，意思是指定当前模型中的哪些layer要被dynamic quantization 
   2. qconfig_spec赋值为一个dict，key为submodule的name或type，value为QConfigDynamic实例（其包含了特定的Observer，比如MinMaxObserver、MovingAverageMinMaxObserver、PerChannelMinMaxObserver、MovingAveragePerChannelMinMaxObserver、HistogramObserver）。
7. 当qconfig_spec为None的时候，quantize_dynamic API就会使用如下的默认值：
   1. `qconfig_spec = {
                   nn.Linear : default_dynamic_qconfig,
                   nn.LSTM : default_dynamic_qconfig,
                   nn.GRU : default_dynamic_qconfig,
                   nn.LSTMCell : default_dynamic_qconfig,
                   nn.RNNCell : default_dynamic_qconfig,
                   nn.GRUCell : default_dynamic_qconfig,
               }`
   2. default_dynamic_qconfig是QConfigDynamic的一个实例，使用如下的参数进行构造：
      1. `default_dynamic_qconfig = QConfigDynamic(activation=default_dynamic_quant_observer, weight=default_weight_observer)`
      2. `default_dynamic_quant_observer = PlaceholderObserver.with_args(dtype=torch.float, compute_dtype=torch.quint8)`
      3. `default_weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)`
      4. 其中，用于activation的PlaceholderObserver 就是个占位符，啥也不做；而用于weight的MinMaxObserver就是记录输入tensor中的最大值和最小值，用来计算scale和zp。