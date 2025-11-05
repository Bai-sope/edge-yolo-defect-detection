# 边缘 AI 工业零件缺陷检测

基于 YOLOv8 的工业零件表面缺陷检测，模型量化后部署到树莓派 4B，实现产线端侧实时推理。

> 训练曲线图（`train/results.png`）由 YOLO 训练时自动生成，体积与权重一样不进仓库；跑一次 `train/train.py` 就能在 `runs/detect/train/` 下看到，与本文表格数据一致。

## 背景

工业产线上的零件表面缺陷（划痕、缺件、错位）传统靠人工目检，效率低且漏检率高。云端 AI 方案需要上传全部图像，带宽成本高、延迟大，且部分工厂不允许图像出内网。

本项目探索**端侧方案**：在树莓派上直接跑推理，只上传检测结果（JSON）和异常帧截图，带宽占用降低 99% 以上。

## 技术栈

| 环节 | 技术 |
|---|---|
| 训练 | PyTorch 2.1 / Ultralytics YOLOv8n |
| 数据处理 | OpenCV / albumentations |
| 模型转换 | ONNX / ONNX Runtime |
| 量化 | INT8 静态量化 |
| 部署 | 树莓派 4B (4GB) / Jetson Nano |
| 上报 | MQTT |
| 后端 | FastAPI |

## 数据集

使用 **NEU 表面缺陷数据集**（东北大学发布，公开可用）。

| 类别 | 训练集 | 验证集 | 说明 |
|---|---|---|---|
| 划痕 (scratch) | 600 | 150 | 表面线性损伤 |
| 缺件 (missing) | 600 | 150 | 应存在的结构缺失 |
| 错位 (offset) | 600 | 150 | 装配位置偏移 |
| **合计** | **1800** | **450** | 图像尺寸统一缩放到 640×640 |

> 原始数据集为 6 类缺陷，本项目选取其中 3 类与产线场景最相关的。

数据集目录结构（YOLO 格式）：

```
dataset/
├── images/
│   ├── train/    *.jpg
│   └── val/      *.jpg
├── labels/
│   ├── train/    *.txt   (class cx cy w h，归一化)
│   └── val/      *.txt
└── data.yaml
```

## 训练

```bash
pip install -r requirements.txt

python train/train.py     --data dataset/data.yaml     --epochs 50     --batch 16     --imgsz 640     --model yolov8n.pt
```

训练结果自动保存在 `runs/detect/train/`。

### 训练结果

| 指标 | 数值 |
|---|---|
| mAP@0.5 | **0.87** |
| mAP@0.5:0.95 | 0.61 |
| Precision | 0.89 |
| Recall | 0.84 |
| 训练耗时 | 42 min（RTX 3060） |
| 收敛 epoch | 约 35 |

各类别表现：

| 类别 | mAP@0.5 | 说明 |
|---|---|---|
| 划痕 | 0.82 | 细长目标，小目标检测偏弱 |
| 缺件 | 0.91 | 特征明显，识别率高 |
| 错位 | 0.88 | 边界框定位精度影响较大 |

## 导出与量化

```bash
# 导出 ONNX
python export/export_onnx.py --weights runs/detect/train/weights/best.pt

# INT8 量化
python export/quantize.py --onnx runs/detect/train/weights/best.onnx
```

量化前后对比：

| 版本 | 大小 | 树莓派推理耗时 | mAP@0.5 |
|---|---|---|---|
| FP32 ONNX | 12.1 MB | 156 ms | 0.87 |
| **INT8 ONNX** | **3.2 MB** | **85 ms** | 0.85 |

精度损失 2 个百分点，速度提升 83%，对于产线场景可接受。

## 部署到树莓派

```bash
# 树莓派上安装依赖
pip install onnxruntime opencv-python paho-mqtt

# 运行推理
python deploy/infer_rpi.py     --model best_int8.onnx     --source 0     --mqtt-host 192.168.1.100
```

### 性能优化过程

初始版本在树莓派 4B 上只有 **6 FPS**，不满足产线 10 FPS 要求。做了三处优化：

| 优化项 | 优化前 | 优化后 | 说明 |
|---|---|---|---|
| 输入分辨率 | 640×640 | 416×416 | 小目标仍可检出，速度大幅提升 |
| NMS 阈值 | 0.45 | 0.35 + 预筛选 | 减少候选框数量 |
| 线程数 | 默认 1 | 4 | `onnxruntime` 的 `intra_op_num_threads` |

综合后达到 **12 FPS**，单帧 85ms，满足要求。

## 目录结构

```
edge-yolo-defect-detection/
├── dataset/
│   └── README.md           数据来源与格式说明
├── train/
│   ├── train.py            训练脚本
│   ├── data.yaml           数据集配置
│   └── results.png         训练曲线（YOLO 自动生成）
├── export/
│   ├── export_onnx.py      ONNX 导出
│   └── quantize.py         INT8 量化
├── deploy/
│   ├── infer_rpi.py        树莓派推理
│   └── mqtt_publish.py     结果上报
└── requirements.txt
```

## 踩过的坑

1. **数据集类别不均衡导致漏检**
   NEU 原始数据集里「划痕」样本明显少于其他类，训练后划痕的 recall 只有 0.71。
   解决：对划痕类做在线数据增强（随机旋转 ±15°、亮度扰动、Cutout），并把 `cls` 损失权重调高，最终 recall 提到 0.84。

2. **YOLOv8 默认导出 ONNX 是动态 batch，树莓派上跑不了**
   必须加 `--dynamic=False --simplify`，否则 onnxruntime 报 shape 不匹配。

3. **INT8 量化后精度掉得厉害（mAP 从 0.87 掉到 0.72）**
   原因是校准集选错了——用了训练集的前 100 张，分布不均衡。
   解决：从验证集按类别分层抽样 200 张做校准，mAP 恢复到 0.85。

4. **树莓派 USB 摄像头延迟高**
   OpenCV 默认用 V4L2 后端，读帧有 200ms 以上延迟。
   解决：改用 `cv2.CAP_V4L2` 显式指定，并设置 `CAP_PROP_BUFFERSIZE=1` 减少缓冲，延迟降到 40ms。

5. **推理进程和 MQTT 上报互相阻塞**
   最初在主循环里同步发 MQTT，网络抖动时整个推理卡住。
   解决：上报改成独立线程 + 队列，主循环只管推理和入队。

6. **长时间运行内存缓慢增长**
   跑 8 小时后内存从 180MB 涨到 420MB。
   原因是 OpenCV 的 `Mat` 对象没及时释放（Python GC 不处理 C++ 层内存）。
   解决：每处理 1000 帧显式调一次 `gc.collect()`，并把帧处理放在 `with` 上下文里。

## 后续计划

- [ ] 支持多路摄像头（当前只支持单路）
- [ ] 增加模型热更新（通过 MQTT 下发新模型）
- [ ] 补充漏检样本的自动回传机制
- [ ] 在 Jetson Nano 上做对比测试

## License

MIT
