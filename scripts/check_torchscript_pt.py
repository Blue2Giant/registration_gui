import argparse
from pathlib import Path
import sys
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_path", type=str, help="TorchScript .pt 文件路径")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    pt_path = Path(args.pt_path)
    if not pt_path.exists():
        print(f"[ERROR] 文件不存在: {pt_path}")
        sys.exit(1)

    device = args.device
    try:
        model = torch.jit.load(str(pt_path), map_location=device)
        model.eval()
        print(f"[OK] torch.jit.load 成功: {pt_path}")
        print(f"[OK] 设备: {device}")
    except Exception as e:
        print(f"[FAIL] torch.jit.load 失败: {pt_path}")
        print(f"[FAIL] 错误: {type(e).__name__}: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
