"""
GPU加速检查工具
用于验证GPU是否正确配置和使用
"""
import torch
import sys

def check_gpu_setup():
    """检查GPU配置和使用情况"""
    
    print("=" * 60)
    print("GPU加速检查工具")
    print("=" * 60)
    
    # 1. 检查CUDA可用性
    print("\n[1] CUDA可用性检查")
    print("-" * 60)
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA是否可用: {cuda_available}")
    
    if not cuda_available:
        print("\n  ⚠️  警告: 未检测到CUDA GPU")
        print("  → 将使用CPU模式（速度较慢）")
        print("  → 如需使用GPU，请安装CUDA和PyTorch GPU版本")
        return False
    
    # 2. GPU信息
    print("\n[2] GPU设备信息")
    print("-" * 60)
    gpu_count = torch.cuda.device_count()
    print(f"  GPU数量: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"\n  GPU {i}:")
        print(f"    名称: {torch.cuda.get_device_name(i)}")
        print(f"    计算能力: {torch.cuda.get_device_capability(i)}")
        
        # 内存信息
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        
        print(f"    总内存: {total_memory:.2f} GB")
        print(f"    已分配: {allocated:.2f} GB")
        print(f"    已保留: {reserved:.2f} GB")
        print(f"    可用: {total_memory - reserved:.2f} GB")
    
    # 3. 当前设备
    print("\n[3] 当前使用设备")
    print("-" * 60)
    current_device = torch.cuda.current_device()
    print(f"  当前GPU: {current_device}")
    print(f"  设备名称: {torch.cuda.get_device_name(current_device)}")
    
    # 4. PyTorch版本信息
    print("\n[4] PyTorch版本信息")
    print("-" * 60)
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA版本: {torch.version.cuda if torch.version.cuda else 'N/A'}")
    print(f"  cuDNN版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    
    # 5. cuDNN设置
    print("\n[5] cuDNN优化设置")
    print("-" * 60)
    print(f"  cuDNN可用: {torch.backends.cudnn.is_available()}")
    print(f"  cuDNN启用: {torch.backends.cudnn.enabled}")
    print(f"  Benchmark模式: {torch.backends.cudnn.benchmark}")
    
    if not torch.backends.cudnn.benchmark:
        print("\n  💡 建议: 启用benchmark模式可以提高性能")
        print("    添加: torch.backends.cudnn.benchmark = True")
    
    # 6. 简单性能测试
    print("\n[6] 简单性能测试")
    print("-" * 60)
    
    try:
        from emonet.models import EmoNet
        
        # 创建测试数据
        test_image = torch.randn(1, 3, 256, 256)
        
        # CPU测试
        device_cpu = 'cpu'
        net_cpu = EmoNet(n_expression=8).to(device_cpu)
        net_cpu.eval()
        
        import time
        
        # 预热
        with torch.no_grad():
            _ = net_cpu(test_image)
        
        # CPU测试
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = net_cpu(test_image)
        cpu_time = (time.time() - start) / 10
        
        # GPU测试
        device_gpu = 'cuda:0'
        net_gpu = EmoNet(n_expression=8).to(device_gpu)
        net_gpu.eval()
        
        # 预热
        with torch.no_grad():
            _ = net_gpu(test_image.to(device_gpu))
        
        # GPU测试
        torch.cuda.synchronize()  # 等待GPU完成
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = net_gpu(test_image.to(device_gpu))
        torch.cuda.synchronize()  # 等待GPU完成
        gpu_time = (time.time() - start) / 10
        
        print(f"  CPU单次推理时间: {cpu_time*1000:.2f} ms")
        print(f"  GPU单次推理时间: {gpu_time*1000:.2f} ms")
        print(f"  加速比: {cpu_time/gpu_time:.2f}x")
        
        if gpu_time < cpu_time:
            print(f"  ✅ GPU加速正常工作！")
        else:
            print(f"  ⚠️  GPU似乎没有加速（可能数据太小或CPU很快）")
            
    except Exception as e:
        print(f"  无法运行性能测试: {e}")
    
    print("\n" + "=" * 60)
    print("检查完成！")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = check_gpu_setup()
    sys.exit(0 if success else 1)


