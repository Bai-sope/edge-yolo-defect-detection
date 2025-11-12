# 数据集说明

## 来源

**NEU 表面缺陷数据集**（NEU Surface Defect Database）
东北大学发布，公开可下载，用于钢材表面缺陷检测研究。

原始数据集包含 6 类缺陷，每类 300 张，共 1800 张，图像尺寸 200×200 灰度图。

## 本项目使用的子集

选取其中 3 类与工业零件表面缺陷最相关的类别，并做了以下处理：

1. **尺寸放大**：200×200 → 640×640（letterbox 填充，保持长宽比）
2. **数据增强**：训练集扩充到 1800 张
   - 随机旋转 ±15°
   - 亮度/对比度扰动 ±20%
   - Cutout（随机遮挡 32×32 区域）
   - 水平翻转
3. **划分**：8:2 分为训练集和验证集

| 类别 | 原始数量 | 增强后（训练） | 验证集 |
|---|---|---|---|
| 划痕 scratch | 300 | 600 | 150 |
| 缺件 missing | 300 | 600 | 150 |
| 错位 offset | 300 | 600 | 150 |
| 合计 | 900 | 1800 | 450 |

## 标注格式

YOLO 格式，每张图对应一个 `.txt` 文件，每行一个目标：

```
<class_id> <cx> <cy> <width> <height>
```

坐标均为归一化值（0~1），相对于图像宽高。

类别映射见 `data.yaml`：

```yaml
path: ../dataset
train: images/train
val: images/val

names:
  0: scratch
  1: missing
  2: offset
```

## 为什么不提交原始图像

1. 数据集体积约 400MB，不适合放 Git 仓库
2. 原数据集有自己的使用条款，转发需遵守

复现方法：从 NEU 官网下载原始数据集，按上述规则处理后放到本目录。

## 校验数据集

```bash
python train/check_dataset.py
```

输出示例：

```
[OK] images/train: 1800 张
[OK] labels/train: 1800 个
[OK] images/val:   450 张
[OK] labels/val:   450 个
[WARN] 发现 3 张图片没有对应标签文件
[WARN] 发现 1 个标签文件为空
```

## 引用

如果使用本数据集，请引用原始论文：

```
Xian Tao, Dapeng Zhang, Wenzhi Ma, Xilong Liu, De Xu.
Automatic Metallic Surface Defect Detection and Recognition with
Convolutional Neural Networks. Applied Sciences, 2018.
```
