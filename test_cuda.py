import torch


def main():
    print("=== PyTorch & CUDA Test ===")

    # PyTorch 版本
    print("PyTorch version:", torch.__version__)

    # CUDA 版本
    print("CUDA version supported by PyTorch:", torch.version.cuda)

    # CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)

    if cuda_available:
        # 当前设备索引
        device = torch.device("cuda:0")
        # 显卡名称
        print("GPU Name:", torch.cuda.get_device_name(device))
        # 测试张量计算
        x = torch.rand(5, 5, device=device)
        y = torch.rand(5, 5, device=device)
        z = x + y
        print("Sample computation on GPU successful:\n", z)
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")
        x = torch.rand(5, 5, device=device)
        y = torch.rand(5, 5, device=device)
        z = x + y
        print("Sample computation on CPU:\n", z)


if __name__ == "__main__":
    main()
