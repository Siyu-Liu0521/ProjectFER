# GPU加速工作原理详解

本文档详细解释 EmoNet 项目中 GPU 加速的实现机制和使用方法。

## 目录
1. [GPU设备检测](#1-gpu设备检测)
2. [模型移动到GPU](#2-模型移动到gpu)
3. [数据移动到GPU](#3-数据移动到gpu)
4. [性能优化技术](#4-性能优化技术)
5. [完整的GPU工作流程](#5-完整的gpu工作流程)
6. [常见问题排查](#6-常见问题排查)

---

## 1. GPU设备检测

### 1.1 自动设备选择

代码使用智能设备选择机制，自动检测是否可用GPU：

```python
# 从 demo.py 和 demo_video.py
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
```

**工作原理：**
- `torch.cuda.is_available()` 检查系统是否有可用的 CUDA GPU
- 如果有GPU，使用 `'cuda:0'`（第一个GPU，索引从0开始）
- 如果没有GPU，自动降级到 `'cpu'`

**示例输出：**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## 2. 模型移动到GPU

### 2.1 模型初始化并移动到GPU

```python
# 从 demo_video.py 的 load_emonet() 函数
def load_emonet(n_expression: int, device: str):
    # 创建模型实例（此时在CPU上）
    net = EmoNet(n_expression=n_expression)
    
    # 将模型移动到指定设备（GPU或CPU）
    net = net.to(device)
    
    # 加载预训练权重
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    net.load_state_dict(state_dict, strict=False)
    
    # 设置为评估模式（禁用dropout等训练时的操作）
    net.eval()
    
    return net
```

**关键点：**
- `.to(device)` 方法会将模型的所有参数和缓冲区（buffers）移动到指定设备
- `map_location='cpu'` 先加载到CPU，然后再移动到GPU，避免GPU内存不足问题
- 模型一旦在GPU上，所有计算都会在GPU上执行

### 2.2 模型内部组件的GPU移动

模型内部的张量也可以直接创建在GPU上：

```python
# 从 emonet.py - 时间平滑的权重直接创建在GPU上
if self.temporal_smoothing:
    self.temporal_weights = torch.Tensor([0.1,0.1,0.15,0.25,0.4])\
        .unsqueeze(0).unsqueeze(2).cuda()  # 直接创建在GPU上
    
    # 在forward中，状态也创建在GPU上
    self.temporal_state = torch.zeros(x.size(0), self.n_temporal_states, 
                                      self.n_expression+self.n_reg).cuda()
```

**注意：** 推荐使用 `.to(device)` 而不是 `.cuda()`，因为前者更灵活（可以轻松切换CPU/GPU）。

---

## 3. 数据移动到GPU

### 3.1 输入数据移动到GPU

在推理时，输入数据必须和模型在同一个设备上：

```python
# 从 demo_video.py 的 run_emonet() 函数
def run_emonet(emonet, frame_rgb):
    # 1. 预处理图像（在CPU上）
    image_rgb = cv2.resize(frame_rgb, (256, 256))
    
    # 2. 转换为张量（在CPU上）
    image_tensor = torch.Tensor(image_rgb).permute(2, 0, 1) / 255.0
    
    # 3. 移动到GPU（关键步骤！）
    image_tensor = image_tensor.to(device)
    
    # 4. 添加batch维度并推理
    with torch.no_grad():
        output = emonet(image_tensor.unsqueeze(0))
```

### 3.2 批量数据处理

在评估脚本中，使用 DataLoader 加载批量数据：

```python
# 从 evaluation.py
for index, data in enumerate(dataloader):
    # 将批量图像移动到GPU
    images = data['image'].to(device)  # 形状: [batch_size, 3, 256, 256]
    
    with torch.no_grad():
        output = net(images)  # 所有计算在GPU上执行
```

**数据传输效率：**
- 批量传输比单张传输更高效
- GPU内存通常有数GB，可以容纳较大的batch
- 默认batch_size=32适合大多数GPU

---

## 4. 性能优化技术

### 4.1 cuDNN Benchmark 优化

```python
# 在所有脚本开头
torch.backends.cudnn.benchmark = True
```

**作用：**
- cuDNN是NVIDIA的深度神经网络库
- `benchmark=True` 会让cuDNN自动寻找最优的卷积算法
- 第一次运行会稍慢（测试不同算法），后续会更快
- **注意：** 只有当输入尺寸固定时才应启用，否则会变慢

**适用场景：**
- ✅ 固定输入尺寸（如256×256）
- ✅ 批量推理
- ❌ 输入尺寸变化频繁

### 4.2 禁用梯度计算

```python
with torch.no_grad():
    output = net(image_tensor)
```

**作用：**
- 推理时不需要计算梯度，禁用可以节省：
  - 约50%的GPU内存
  - 约30-50%的计算时间
- 这是推理时的标准做法

### 4.3 CPU-GPU数据传输最小化

**优化原则：**
1. **批量处理**：一次传输多张图像，而不是逐张传输
2. **延迟传输**：只在需要时传输到GPU
3. **就地操作**：尽可能在GPU上完成所有操作

**示例（高效）：**
```python
# 在GPU上完成所有操作
images = data['image'].to(device)  # 批量传输
output = net(images)  # GPU计算
results = output.cpu().numpy()  # 最后才传回CPU
```

**示例（低效）：**
```python
# 频繁在CPU和GPU之间传输
for img in images:
    img_gpu = img.to(device)  # 每次单独传输
    output = net(img_gpu)
    result = output.cpu()  # 每次传回CPU
```

---

## 5. 完整的GPU工作流程

### 5.1 视频处理的GPU工作流程

```
初始化阶段:
├─ 检测GPU可用性 → device = 'cuda:0' or 'cpu'
├─ 加载EmoNet模型 → net.to(device)
├─ 加载SFD检测器 → SFDDetector(device)
└─ 启用cuDNN benchmark

对每一帧:
├─ 1. 人脸检测（可能在GPU上，取决于SFD实现）
│   └─ 返回边界框（CPU上的numpy数组）
│
├─ 2. 裁剪人脸（CPU操作）
│   └─ face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
│
├─ 3. 预处理（CPU操作）
│   ├─ 调整大小: resize to 256×256
│   └─ 转换为张量: torch.Tensor(...)
│
├─ 4. 数据传输: CPU → GPU
│   └─ image_tensor = image_tensor.to(device)
│
├─ 5. GPU推理（GPU操作）
│   ├─ with torch.no_grad():  # 禁用梯度
│   └─ output = emonet(image_tensor)
│
├─ 6. 结果处理（部分在GPU，部分在CPU）
│   ├─ 置信度计算: emotion_probs = softmax(output["expression"])
│   ├─ 质量计算: quality = sqrt(confidence * heatmap_quality)
│   └─ 提取值: confidence.cpu().item()
│
└─ 7. 可视化（CPU操作）
    └─ 在原始帧上绘制结果
```

### 5.2 GPU内存管理

**典型GPU内存使用：**
- EmoNet模型：约 200-300 MB
- SFD检测器：约 100-200 MB
- 单张图像（256×256×3）：约 0.75 MB
- Batch（32张图像）：约 24 MB
- 中间特征图：约 100-500 MB（取决于batch size）

**总内存需求：**
- 推理：约 1-2 GB GPU内存
- 训练：约 4-8 GB GPU内存（取决于batch size）

---

## 6. 常见问题排查

### 6.1 检查GPU是否被使用

```python
import torch

# 检查CUDA可用性
print(f"CUDA available: {torch.cuda.is_available()}")

# 检查GPU数量
print(f"GPU count: {torch.cuda.device_count()}")

# 检查当前设备
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # 检查GPU内存
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
```

### 6.2 检查模型和设备是否匹配

```python
# 检查模型参数所在的设备
for name, param in net.named_parameters():
    print(f"{name}: {param.device}")
    break  # 只打印第一个参数作为示例

# 检查输入数据所在设备
print(f"Input tensor device: {image_tensor.device}")

# 确保模型和数据在同一设备
assert next(net.parameters()).device == image_tensor.device
```

### 6.3 常见错误及解决方法

**错误1: RuntimeError: Input type and weight type should be the same**
```python
# 原因：模型在GPU上，但输入在CPU上（或反之）
# 解决：确保数据和模型在同一设备
image_tensor = image_tensor.to(device)  # 添加这一行
```

**错误2: CUDA out of memory**
```python
# 原因：GPU内存不足
# 解决方法：
# 1. 减小batch size
batch_size = 16  # 从32减小到16

# 2. 清理GPU缓存
torch.cuda.empty_cache()

# 3. 使用CPU（作为备选）
device = 'cpu'
```

**错误3: 模型太慢（没有使用GPU加速）**
```python
# 检查设备
print(f"Using device: {device}")

# 检查是否真的在GPU上
if next(net.parameters()).is_cuda:
    print("Model is on GPU ✓")
else:
    print("Model is on CPU ✗")
```

### 6.4 性能对比测试

```python
import time

# CPU测试
device_cpu = 'cpu'
net_cpu = EmoNet(n_expression=8).to(device_cpu)
net_cpu.load_state_dict(state_dict, strict=False)
net_cpu.eval()

start = time.time()
with torch.no_grad():
    output = net_cpu(image_tensor.to(device_cpu))
cpu_time = time.time() - start

# GPU测试
device_gpu = 'cuda:0'
net_gpu = EmoNet(n_expression=8).to(device_gpu)
net_gpu.load_state_dict(state_dict, strict=False)
net_gpu.eval()

start = time.time()
with torch.no_grad():
    output = net_gpu(image_tensor.to(device_gpu))
gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.4f}s")
print(f"GPU time: {gpu_time:.4f}s")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")
```

**预期结果：**
- GPU通常比CPU快 10-100倍（取决于GPU型号和模型复杂度）
- 批量处理时，GPU优势更明显

---

## 总结

GPU加速的关键步骤：
1. ✅ 自动检测GPU可用性
2. ✅ 将模型移动到GPU: `net.to(device)`
3. ✅ 将输入数据移动到GPU: `data.to(device)`
4. ✅ 使用 `torch.no_grad()` 禁用梯度
5. ✅ 启用 `cudnn.benchmark` 优化
6. ✅ 批量处理以提高效率

通过这些优化，GPU可以显著加速深度学习推理过程！


