#!/usr/bin/env python3
"""
MQTT 上报工具（可独立测试）

用法:
  python deploy/mqtt_publish.py --host 192.168.1.100 --device rpi-01 --count 3
"""
import argparse
import json
import random
import sys
import time

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print('[!] 需要 paho-mqtt: pip install paho-mqtt')
    sys.exit(1)

CLASSES = ['scratch', 'missing', 'offset']


def main():
    p = argparse.ArgumentParser(description='MQTT 上报测试')
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, default=1883)
    p.add_argument('--device', default='rpi-01')
    p.add_argument('--count', type=int, default=1, help='发送几条')
    p.add_argument('--interval', type=float, default=1.0)
    args = p.parse_args()

    topic = f'iot/pk2/{args.device}/up'
    client = mqtt.Client(client_id=f'test-{args.device}')

    print(f'[*] 连接 {args.host}:{args.port} ...')
    try:
        client.connect(args.host, args.port, keepalive=60)
    except Exception as e:
        print(f'[!] 连接失败: {e}')
        sys.exit(1)

    client.loop_start()

    for i in range(args.count):
        n_det = random.randint(1, 3)
        detections = []
        for _ in range(n_det):
            detections.append({
                'cls': random.choice(CLASSES),
                'conf': round(random.uniform(0.4, 0.95), 3),
                'box': [
                    round(random.uniform(0, 300), 1),
                    round(random.uniform(0, 200), 1),
                    round(random.uniform(300, 640), 1),
                    round(random.uniform(200, 480), 1),
                ]
            })

        payload = {
            'deviceId': args.device,
            'ts': int(time.time() * 1000),
            'count': len(detections),
            'detections': detections,
        }

        info = client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=1)
        info.wait_for_publish(timeout=5)
        print(f'[{i+1}/{args.count}] -> {topic}')
        print(f'    {json.dumps(payload, ensure_ascii=False)[:120]}...')

        if i < args.count - 1:
            time.sleep(args.interval)

    client.loop_stop()
    client.disconnect()
    print('[*] 完成')


if __name__ == '__main__':
    main()
