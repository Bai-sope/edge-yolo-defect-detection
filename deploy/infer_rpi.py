#!/usr/bin/env python3
"""
树莓派端侧推理

优化点（相对初版 6 FPS → 12 FPS）：
  1. 输入分辨率 640 → 416
  2. NMS 阈值调整 + 候选框预筛选
  3. onnxruntime 多线程

用法:
  python deploy/infer_rpi.py --model best_int8.onnx --source 0 --mqtt-host 192.168.1.100
"""
import argparse
import json
import os
import sys
import time
import queue
import threading
import gc

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print('[!] 需要 onnxruntime: pip install onnxruntime')
    sys.exit(1)

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


CLASS_NAMES = ['scratch', 'missing', 'offset']


class YoloOnnx:
    """ONNX Runtime 推理封装"""

    def __init__(self, model_path, imgsz=416, conf_thres=0.35, iou_thres=0.35, threads=4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        # 树莓派上设线程数能明显提速
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def preprocess(self, frame):
        """BGR -> RGB, resize, normalize, NCHW"""
        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.ascontiguousarray(np.expand_dims(img, 0))

    def infer(self, frame):
        """返回 [(class_id, conf, x1, y1, x2, y2), ...]"""
        inp = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: inp})
        return self._postprocess(outputs[0], frame.shape)

    def _postprocess(self, pred, orig_shape):
        """
        YOLOv8 输出格式: (1, 4+nc, 8400)
        前 4 行是 cx, cy, w, h，后面是各类别的分数
        """
        pred = pred[0].T   # -> (8400, 4+nc)

        # 类别分数取最大值
        scores = pred[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # 置信度过滤（先粗筛，减少后续 NMS 计算量）
        mask = confidences > self.conf_thres
        if not np.any(mask):
            return []

        boxes = pred[mask, :4]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # cxcywh -> xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # 缩放到原图尺寸
        h, w = orig_shape[:2]
        boxes_xyxy[:, [0, 2]] *= w / self.imgsz
        boxes_xyxy[:, [1, 3]] *= h / self.imgsz

        # NMS
        indices = self._nms(boxes_xyxy, confidences, self.iou_thres)

        results = []
        for i in indices:
            results.append((
                int(class_ids[i]),
                float(confidences[i]),
                *boxes_xyxy[i].tolist()
            ))
        return results

    @staticmethod
    def _nms(boxes, scores, iou_thres):
        """纯 numpy 实现的 NMS"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]

        return keep


class MqttReporter(threading.Thread):
    """
    MQTT 上报线程

    为什么要独立线程：网络抖动时同步发 MQTT 会阻塞主推理循环。
    实测不加线程时，网络差的情况下 FPS 会从 12 掉到 3。
    """

    def __init__(self, host, port, topic, device_id, queue_size=100):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.topic = topic
        self.device_id = device_id
        self.queue = queue.Queue(maxsize=queue_size)
        self.client = None
        self.running = True

    def connect(self):
        if mqtt is None:
            print('[!] paho-mqtt 未安装，跳过上报')
            return False
        try:
            self.client = mqtt.Client(client_id=f'rpi-{self.device_id}')
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
            print(f'[mqtt] connected to {self.host}:{self.port}')
            return True
        except Exception as e:
            print(f'[!] MQTT 连接失败: {e}')
            return False

    def submit(self, detections):
        """非阻塞入队，满了就丢最旧的"""
        payload = {
            'deviceId': self.device_id,
            'ts': int(time.time() * 1000),
            'count': len(detections),
            'detections': detections,
        }
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(payload)
            except queue.Empty:
                pass

    def run(self):
        while self.running:
            try:
                payload = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self.client:
                try:
                    self.client.publish(
                        self.topic,
                        json.dumps(payload, ensure_ascii=False),
                        qos=1
                    )
                except Exception as e:
                    print(f'[mqtt] publish failed: {e}')

    def stop(self):
        self.running = False
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


def draw_results(frame, detections):
    """在帧上画检测框"""
    for cid, conf, x1, y1, x2, y2 in detections:
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
        color = [(0, 0, 255), (0, 255, 0), (255, 128, 0)][cid % 3]

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f'{name} {conf:.2f}'
        cv2.putText(frame, label, (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def main():
    p = argparse.ArgumentParser(description='树莓派端侧推理')
    p.add_argument('--model', default='best_int8.onnx')
    p.add_argument('--source', default='0', help='摄像头索引或视频路径')
    p.add_argument('--imgsz', type=int, default=416)
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--iou', type=float, default=0.35)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--mqtt-host', default=None, help='不填则不上报')
    p.add_argument('--mqtt-port', type=int, default=1883)
    p.add_argument('--device-id', default='rpi-01')
    p.add_argument('--show', action='store_true', help='显示窗口（无显示器时不要开）')
    p.add_argument('--save-dir', default=None, help='异常帧保存目录')
    args = p.parse_args()

    print('=' * 60)
    print('  边缘 AI 缺陷检测 —— 树莓派推理')
    print('=' * 60)

    model = YoloOnnx(args.model, args.imgsz, args.conf, args.iou, args.threads)
    print(f'[+] 模型加载: {args.model}')

    # 摄像头：显式指定 V4L2 后端，减少缓冲延迟
    src = int(args.source) if args.source.isdigit() else args.source
    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # 关键：减少缓冲
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f'[!] 无法打开视频源: {args.source}')
        sys.exit(1)
    print(f'[+] 视频源已打开: {args.source}')

    reporter = None
    if args.mqtt_host:
        reporter = MqttReporter(
            args.mqtt_host, args.mqtt_port,
            f'iot/pk2/{args.device_id}/up',
            args.device_id
        )
        if reporter.connect():
            reporter.start()
        else:
            reporter = None

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    frame_count = 0
    fps_t0 = time.time()
    gc_interval = 1000

    print('[+] 开始推理，Ctrl+C 退出')
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('[!] 读帧失败')
                break

            t0 = time.time()
            detections = model.infer(frame)
            infer_ms = (time.time() - t0) * 1000

            frame_count += 1

            # 每 30 帧打印一次 FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_t0
                fps = 30 / elapsed if elapsed > 0 else 0
                print(f'[stat] frame={frame_count} '
                      f'fps={fps:.1f} infer={infer_ms:.0f}ms '
                      f'detections={len(detections)}')
                fps_t0 = time.time()

            # 上报
            if reporter and detections:
                det_list = [
                    {'cls': CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c),
                     'conf': round(conf, 3),
                     'box': [round(v, 1) for v in (x1, y1, x2, y2)]}
                    for c, conf, x1, y1, x2, y2 in detections
                ]
                reporter.submit(det_list)

            # 异常帧存档
            if args.save_dir and detections:
                path = os.path.join(args.save_dir, f'frame_{frame_count:06d}.jpg')
                cv2.imwrite(path, frame)

            if args.show:
                vis = draw_results(frame.copy(), detections)
                cv2.imshow('defect detection', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 定期 GC，防止 OpenCV 的 C++ 内存泄漏累积
            if frame_count % gc_interval == 0:
                gc.collect()

    except KeyboardInterrupt:
        print('[*] 用户中断')
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        if reporter:
            reporter.stop()
        print(f'[*] 共处理 {frame_count} 帧')


if __name__ == '__main__':
    main()
