import torch

def test_torch_installation():
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.get_device_name(0)}")
        
    # 测试GPU计算
    if torch.cuda.is_available():
        # 创建一个张量并移动到GPU
        x = torch.randn(2, 3).cuda()
        print("\nGPU测试张量:")
        print(x)
        print(f"张量设备: {x.device}")
    else:
        print("\nGPU不可用,只能使用CPU")

if __name__ == "__main__":
    test_torch_installation() 