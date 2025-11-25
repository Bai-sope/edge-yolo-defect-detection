#!/usr/bin/env python3
"""
YOLOv8 缺陷检测训练脚本

用法:
  python train/train.py --data train/data.yaml --epochs 50 --batch 16
"""
import argparse
import os
import sys

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description='YOLOv8 缺陷检测训练')
    p.add_argument('--data', default='train/data.yaml', help='数据集配置')
    p.add_argument('--model', default='yolov8n.pt', help='预训练权重')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0', help="'0' 用 GPU，'cpu' 用 CPU")
    p.add_argument('--project', default='runs/detect', help='输出目录')
    p.add_argument('--name', default='train', help='实验名')
    p.add_argument('--resume', action='store_true', help='从上次中断处继续')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        print(f'[!] 数据集配置不存在: {args.data}')
        print('    请先按 dataset/README.md 准备数据')
        sys.exit(1)

    print('=' * 60)
    print('  YOLOv8 缺陷检测训练')
    print('=' * 60)
    print(f'  数据集:   {args.data}')
    print(f'  预训练:   {args.model}')
    print(f'  epochs:   {args.epochs}')
    print(f'  batch:    {args.batch}')
    print(f'  imgsz:    {args.imgsz}')
    print(f'  device:   {args.device}')
    print('=' * 60)
    print()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,

        # 优化器与学习率
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,

        # 数据增强
        # 注意：划痕类样本少，翻转和旋转是必要的
        degrees=15.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        flipud=0.0,        # 缺陷有方向性，不做上下翻转
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,

        # 训练策略
        patience=15,       # 15 轮不提升就早停
        save=True,
        save_period=10,
        val=True,
        plots=True,        # 生成混淆矩阵、PR曲线等图

        # 日志
        verbose=True,
        seed=42,
    )

    print()
    print('=' * 60)
    print('  训练完成')
    print('=' * 60)
    print(f'  最佳权重: {args.project}/{args.name}/weights/best.pt')
    print(f'  训练曲线: {args.project}/{args.name}/results.png')
    print(f'  混淆矩阵: {args.project}/{args.name}/confusion_matrix.png')
    print()
    print('  下一步：')
    print(f'    python export/export_onnx.py --weights {args.project}/{args.name}/weights/best.pt')

    # 打印最终指标
    try:
        metrics = results.results_dict
        print()
        print('  最终指标：')
        for k, v in metrics.items():
            print(f'    {k}: {v:.4f}')
    except Exception:
        pass


if __name__ == '__main__':
    main()
