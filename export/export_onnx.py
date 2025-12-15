#!/usr/bin/env python3
"""
导出 ONNX 模型

注意：树莓派上跑必须用 static shape + simplify，
     否则 onnxruntime 会因为动态维度报错。
"""
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description='导出 YOLOv8 为 ONNX')
    p.add_argument('--weights', default='runs/detect/train/weights/best.pt')
    p.add_argument('--imgsz', type=int, default=416,
                   help='输入尺寸。树莓派建议 416 或 320')
    p.add_argument('--output', default=None, help='输出路径')
    args = p.parse_args()

    if not os.path.exists(args.weights):
        print(f'[!] 权重文件不存在: {args.weights}')
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print('[!] 需要 ultralytics: pip install ultralytics')
        sys.exit(1)

    print('=' * 60)
    print('  导出 ONNX')
    print('=' * 60)
    print(f'  权重:   {args.weights}')
    print(f'  imgsz:  {args.imgsz}')
    print('=' * 60)
    print()

    model = YOLO(args.weights)

    # dynamic=False 是关键：树莓派上 onnxruntime 不支持动态 shape
    # simplify=True 会做图优化，减少冗余算子
    path = model.export(
        format='onnx',
        imgsz=args.imgsz,
        dynamic=False,
        simplify=True,
        opset=12,          # opset 12 兼容性最好
        half=False,        # FP16 在 ARM 上收益不大
    )

    print()
    print(f'[+] 导出成功: {path}')
    print(f'    大小: {os.path.getsize(path) / 1024 / 1024:.1f} MB')
    print()
    print('  下一步（可选，量化）：')
    print(f'    python export/quantize.py --onnx {path}')


if __name__ == '__main__':
    main()
