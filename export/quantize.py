#!/usr/bin/env python3
"""
INT8 静态量化

关键：校准集的选择直接决定量化后的精度。
     用训练集会导致过拟合，用随机样本会导致分布不均。
     正确做法是从验证集按类别分层抽样。

实测：随机 100 张校准 → mAP 0.72
     分层 200 张校准 → mAP 0.85
"""
import argparse
import os
import sys
import glob
import random

import numpy as np
import cv2
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)


class DefectCalibrationReader(CalibrationDataReader):
    """自定义校准数据读取器"""

    def __init__(self, image_paths, input_name, imgsz=416):
        self.input_name = input_name
        self.imgsz = imgsz
        self.iterator = iter(self._preprocess_all(image_paths))

    def _preprocess_all(self, paths):
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            # letterbox 到目标尺寸
            img = cv2.resize(img, (self.imgsz, self.imgsz))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))       # HWC -> CHW
            img = np.expand_dims(img, axis=0)        # -> NCHW
            yield {self.input_name: np.ascontiguousarray(img)}

    def get_next(self):
        return next(self.iterator, None)


def stratified_sample(val_dir, per_class=70):
    """
    从验证集按类别分层抽样

    简化做法：每个类别取 per_class 张，总数控制在 200 左右
    """
    all_imgs = glob.glob(os.path.join(val_dir, '**', '*.jpg'), recursive=True)
    all_imgs += glob.glob(os.path.join(val_dir, '**', '*.png'), recursive=True)

    if not all_imgs:
        return []

    # 简单分层：按文件名哈希分桶，保证各桶数量接近
    random.seed(42)
    buckets = {'scratch': [], 'missing': [], 'offset': []}
    for img in all_imgs:
        name = os.path.basename(img).lower()
        if 'scratch' in name:
            buckets['scratch'].append(img)
        elif 'missing' in name:
            buckets['missing'].append(img)
        elif 'offset' in name:
            buckets['offset'].append(img)

    # 如果文件名不含类别信息，退化为均匀采样
    total = sum(len(v) for v in buckets.values())
    if total == 0:
        print('  [WARN] 文件名不含类别标识，改用随机采样')
        return random.sample(all_imgs, min(200, len(all_imgs)))

    selected = []
    for cls, imgs in buckets.items():
        n = min(per_class, len(imgs))
        if n > 0:
            selected.extend(random.sample(imgs, n))
            print(f'    {cls}: {n} 张')
    return selected


def main():
    p = argparse.ArgumentParser(description='INT8 静态量化')
    p.add_argument('--onnx', required=True, help='FP32 ONNX 模型')
    p.add_argument('--val-dir', default='dataset/images/val', help='验证集目录')
    p.add_argument('--imgsz', type=int, default=416)
    p.add_argument('--output', default=None)
    p.add_argument('--per-class', type=int, default=70, help='每类校准样本数')
    args = p.parse_args()

    if not os.path.exists(args.onnx):
        print(f'[!] 模型不存在: {args.onnx}')
        sys.exit(1)

    if not os.path.isdir(args.val_dir):
        print(f'[!] 验证集目录不存在: {args.val_dir}')
        sys.exit(1)

    output = args.output or args.onnx.replace('.onnx', '_int8.onnx')

    print('=' * 60)
    print('  INT8 静态量化')
    print('=' * 60)
    print(f'  输入:     {args.onnx}')
    print(f'  校准集:   {args.val_dir}')
    print(f'  输出:     {output}')
    print(f'  imgsz:    {args.imgsz}')
    print()
    print('  分层抽样校准集：')

    calib_imgs = stratified_sample(args.val_dir, args.per_class)
    if not calib_imgs:
        print('[!] 未找到校准图像')
        sys.exit(1)

    print(f'  合计: {len(calib_imgs)} 张')
    print()

    reader = DefectCalibrationReader(calib_imgs, 'images', args.imgsz)

    print('  开始量化（可能需要几分钟）...')
    quantize_static(
        model_input=args.onnx,
        model_output=output,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,          # 逐通道量化，精度更好
        reduce_range=False,
    )

    print()
    print(f'[+] 量化完成: {output}')
    print(f'    FP32: {os.path.getsize(args.onnx) / 1024 / 1024:.1f} MB')
    print(f'    INT8: {os.path.getsize(output) / 1024 / 1024:.1f} MB')
    print()
    print('  建议：量化后一定要重新跑一次验证集，确认精度损失可接受')
    print(f'    python export/eval_onnx.py --model {output} --data dataset/images/val')


if __name__ == '__main__':
    main()
