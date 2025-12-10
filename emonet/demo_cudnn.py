"""
cuDNN Benchmark模式演示脚本
展示benchmark模式如何自动选择最优算法并加速推理
"""
import torch
import time
import sys
from pathlib import Path

def demo_cudnn_benchmark():
    """演示cuDNN benchmark模式的效果"""
    
    print("=" * 70)
    print("cuDNN Benchmark模式演示")
    print("=" * 70)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("\n❌ 错误: 未检测到CUDA GPU")
        print("此演示需要GPU才能运行")
        return
    
    # 检查cuDNN可用性
    if not torch.backends.cudnn.is_available():
        print("\n❌ 错误: cuDNN不可用")
        print("请确保安装了cuDNN")
        return
    
    print(f"\n✅ CUDA可用: {torch.cuda.get_device_name(0)}")
    print(f"✅ cuDNN可用: 版本 {torch.backends.cudnn.version()}")
    
    # 创建测试模型（使用EmoNet或简单的卷积网络）
    try:
        from emonet.models import EmoNet
        print("\n使用EmoNet模型进行测试...")
        net = EmoNet(n_expression=8).cuda()
        net.eval()
        test_input = torch.randn(1, 3, 256, 256).cuda()
        model_name = "EmoNet"
    except:
        print("\n使用简单的卷积网络进行测试...")
        # 创建一个简单的卷积网络
        net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 8)
        ).cuda()
        net.eval()
        test_input = torch.randn(1, 3, 256, 256).cuda()
        model_name = "SimpleConvNet"
    
    num_iterations = 50
    
    print(f"\n{'='*70}")
    print("测试1: 禁用Benchmark模式")
    print(f"{'='*70}")
    
    # 禁用benchmark
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True  # 确保cuDNN启用
    
    # 预热（让GPU稳定）
    print("预热GPU...")
    with torch.no_grad():
        for _ in range(5):
            _ = net(test_input)
    torch.cuda.synchronize()
    
    # 测试性能
    print(f"运行 {num_iterations} 次推理...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = net(test_input)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    avg_time_no_benchmark = elapsed_time / num_iterations
    
    print(f"\n结果:")
    print(f"  总时间: {elapsed_time:.3f} 秒")
    print(f"  平均每次推理: {avg_time_no_benchmark*1000:.2f} ms")
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    time.sleep(1)  # 等待GPU稳定
    
    print(f"\n{'='*70}")
    print("测试2: 启用Benchmark模式")
    print(f"{'='*70}")
    
    # 启用benchmark
    torch.backends.cudnn.benchmark = True
    
    # 第一次推理（会测试算法，较慢）
    print("第一次推理（算法测试阶段，可能较慢）...")
    torch.cuda.synchronize()
    start_first = time.time()
    with torch.no_grad():
        _ = net(test_input)
    torch.cuda.synchronize()
    first_time = time.time() - start_first
    
    print(f"  第一次推理时间: {first_time*1000:.2f} ms")
    print(f"  (包含算法测试和选择)")
    
    # 预热（使用缓存的最优算法）
    print("\n预热GPU（使用缓存的最优算法）...")
    with torch.no_grad():
        for _ in range(5):
            _ = net(test_input)
    torch.cuda.synchronize()
    
    # 测试性能
    print(f"运行 {num_iterations} 次推理...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = net(test_input)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    avg_time_with_benchmark = elapsed_time / num_iterations
    
    print(f"\n结果:")
    print(f"  总时间: {elapsed_time:.3f} 秒")
    print(f"  平均每次推理: {avg_time_with_benchmark*1000:.2f} ms")
    
    # 比较结果
    print(f"\n{'='*70}")
    print("性能对比")
    print(f"{'='*70}")
    print(f"模型: {model_name}")
    print(f"输入尺寸: {list(test_input.shape)}")
    print(f"\n{'指标':<25} {'禁用Benchmark':<20} {'启用Benchmark':<20}")
    print("-" * 70)
    print(f"{'平均推理时间 (ms)':<25} {avg_time_no_benchmark*1000:>18.2f} {avg_time_with_benchmark*1000:>18.2f}")
    
    speedup = avg_time_no_benchmark / avg_time_with_benchmark
    print(f"{'加速比':<25} {'-':<20} {speedup:>18.2f}x")
    
    time_saved = (avg_time_no_benchmark - avg_time_with_benchmark) * 1000
    print(f"{'每次节省时间 (ms)':<25} {'-':<20} {time_saved:>18.2f}")
    
    # 结论
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    if speedup > 1.1:
        print(f"✅ Benchmark模式提供了 {speedup:.2f}x 的加速！")
        print("  这意味着cuDNN自动选择了更优的算法。")
    elif speedup > 0.9:
        print(f"⚠️  Benchmark模式效果不明显（{speedup:.2f}x）")
        print("  可能原因：")
        print("    - 输入尺寸较小，算法差异不明显")
        print("    - 当前GPU已经选择了较优算法")
        print("    - 模型简单，优化空间有限")
    else:
        print(f"❌ Benchmark模式似乎变慢了（{speedup:.2f}x）")
        print("  这不应该发生，可能的原因：")
        print("    - GPU状态不稳定")
        print("    - 测试时间过短，结果不准确")
        print("    - 系统负载影响")
    
    print(f"\n{'='*70}")
    print("cuDNN配置信息")
    print(f"{'='*70}")
    print(f"  cuDNN可用: {torch.backends.cudnn.is_available()}")
    print(f"  cuDNN启用: {torch.backends.cudnn.enabled}")
    print(f"  Benchmark模式: {torch.backends.cudnn.benchmark}")
    print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"  确定性模式: {torch.backends.cudnn.deterministic}")
    
    print(f"\n{'='*70}")
    print("建议")
    print(f"{'='*70}")
    if avg_time_no_benchmark > avg_time_with_benchmark:
        print("✅ 建议启用benchmark模式以获得最佳性能")
        print("   在代码开头添加:")
        print("   torch.backends.cudnn.benchmark = True")
    else:
        print("⚠️  Benchmark模式效果不明显")
        print("   可能原因：输入尺寸变化频繁或模型简单")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        demo_cudnn_benchmark()
    except KeyboardInterrupt:
        print("\n\n中断测试")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


