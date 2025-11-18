#!/usr/bin/env python3
"""
数据集完整性检查

检查项：
  1. 图片和标签是否一一对应
  2. 标签文件格式是否正确（5 个字段、归一化范围 0~1）
  3. 类别 ID 是否在 nc 范围内
  4. 空标签文件、无标签图片

用法:
  python train/check_dataset.py
"""
import os
import sys

try:
    import yaml
except ImportError:
    print('[!] 需要 pyyaml: pip install pyyaml')
    sys.exit(1)

DATASET_ROOT = os.path.join(os.path.dirname(__file__), '..', 'dataset')
DATA_YAML = os.path.join(os.path.dirname(__file__), 'data.yaml')

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def load_config():
    with open(DATA_YAML, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def check_split(split_name, rel_path, nc):
    img_dir = os.path.join(DATASET_ROOT, rel_path)
    lbl_dir = img_dir.replace('images', 'labels')

    if not os.path.isdir(img_dir):
        print(f'[!] 图片目录不存在: {img_dir}')
        return None

    imgs = {os.path.splitext(f)[0] for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)
            if f.endswith('.txt')} if os.path.isdir(lbl_dir) else set()

    print(f'[OK] {rel_path}: {len(imgs)} 张图')
    print(f'[OK] labels: {len(lbls)} 个标签')

    no_label = imgs - lbls
    no_img = lbls - imgs
    if no_label:
        print(f'[WARN] {len(no_label)} 张图片没有标签文件')
        for n in list(no_label)[:5]:
            print(f'       - {n}')
    if no_img:
        print(f'[WARN] {len(no_img)} 个标签没有对应图片')
        for n in list(no_img)[:5]:
            print(f'       - {n}')

    # 校验标签内容
    bad_format = []
    empty = []
    bad_class = []
    out_of_range = []

    for name in lbls:
        path = os.path.join(lbl_dir, name + '.txt')
        try:
            with open(path, encoding='utf-8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except OSError:
            continue

        if not lines:
            empty.append(name)
            continue

        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                bad_format.append((name, ln))
                continue
            try:
                cid = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                bad_format.append((name, ln))
                continue

            if cid < 0 or cid >= nc:
                bad_class.append((name, cid))
            if any(c < 0 or c > 1 for c in coords):
                out_of_range.append((name, ln))

    if empty:
        print(f'[WARN] {len(empty)} 个标签文件为空')
        for n in empty[:5]:
            print(f'       - {n}.txt')
    if bad_format:
        print(f'[ERR]  {len(bad_format)} 行格式错误（应为 5 个字段）')
        for n, ln in bad_format[:5]:
            print(f'       - {n}: {ln}')
    if bad_class:
        print(f'[ERR]  {len(bad_class)} 处类别 ID 越界（nc={nc}）')
        for n, c in bad_class[:5]:
            print(f'       - {n}: class={c}')
    if out_of_range:
        print(f'[ERR]  {len(out_of_range)} 处坐标超出 0~1 范围')
        for n, ln in out_of_range[:5]:
            print(f'       - {n}: {ln}')

    return {
        'imgs': len(imgs),
        'lbls': len(lbls),
        'errors': len(bad_format) + len(bad_class) + len(out_of_range),
    }


def main():
    print('=' * 60)
    print('  数据集检查')
    print('=' * 60)

    cfg = load_config()
    nc = cfg.get('nc', 3)
    names = cfg.get('names', {})
    print(f'  类别数: {nc}')
    for k, v in names.items():
        print(f'    {k}: {v}')
    print()

    for split, rel in [('train', cfg['train']), ('val', cfg['val'])]:
        print(f'--- {split} ---')
        check_split(split, rel, nc)
        print()

    print('=' * 60)
    print('  检查完成')


if __name__ == '__main__':
    main()
