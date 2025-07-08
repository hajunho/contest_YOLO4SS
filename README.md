# 🎣 넝마주의 with YOLOv8

해양 쓰레기 탐지를 위한 YOLOv8 기반 객체 탐지 모델입니다.

## 📥 Model Download
[**Download best.pt**](https://drive.google.com/file/d/1sNWl2KNQM7zdf55uHZ9g-tz6svh-280Q/view?usp=sharing)

## 🧱 데이터 구성
- **이미지**: `./images/` 폴더에 .jpg 파일 (예: tire_045_00123.jpg)
- **라벨 (VOC 형식)**: `./annotations/` 폴더에 .xml 파일

## 🧾 클래스 정의 (총 10종)

```python
class_map = {
    "tire": 0,
    "wood": 1,
    "rope": 2,
    "spring fish trap": 3,
    "bundle of ropes": 4,
    "circular fish trap": 5,
    "eel fish trap": 6,
    "fish net": 7,
    "rectangular fish trap": 8,
    "other objects": 9
}
```

## 🚀 Quick Setup

### 필요 패키지 설치
```bash
pip install ultralytics opencv-python torch torchvision typing_extensions pyyaml
```

### 안전한 가상환경 사용
```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install ultralytics opencv-python torch torchvision
```

## 🔄 데이터 전처리

### 1. XML → YOLO .txt 변환
```bash
python3 convert_voc_to_yolotxt.py
```
- `.txt` 파일은 `./labels_yolo/`에 생성됨
- YOLO 포맷: `class_id x_center y_center width height`

### 2. Train/Val 셋 분리
```bash
python3 split_testNval.py
```
- `dataset/images/train`, `dataset/images/val` 등으로 정리됨

### 3. data.yaml 설정
```yaml
path: /home/YOLO/dataset
train: images/train
val: images/val
names:
  0: tire
  1: wood
  2: rope
  3: spring fish trap
  4: bundle of ropes
  5: circular fish trap
  6: eel fish trap
  7: fish net
  8: rectangular fish trap
  9: other objects
```

## 🧠 모델 학습

### YOLOv8 학습 실행
```bash
yolo detect train \
  model=yolov8m.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=640 \
  device=0
```
- `best.pt` 결과는 `runs/detect/train/weights/`에 저장됨

## 🎥 영상 추론

### 기본 사용법
```bash
python3 test01.py  # 또는 test01_debug.py
```
- 입력: `seetrash_val_01.mp4`
- 출력: `seetrash_val_01_detected.mp4`
- YOLO 추론 결과를 프레임 단위로 영상에 박스로 시각화

### 영상 처리 예제 코드
```python
import cv2
from ultralytics import YOLO

model_path = './best.pt'
video_path = './seetrash_val_01.mp4'
output_path = './seetrash_val_01_detected.mp4'

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 입력 영상 열기 실패")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"🎥 영상 정보: {width}x{height} @ {fps}fps")

if fps == 0.0:
    fps = 24.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
detected_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame, verbose=False)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        detected_count += 1
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)  # 빨간색, 굵기 4

    out.write(frame)

cap.release()
out.release()

print(f"[✅ 완료] {frame_count}프레임 중 {detected_count}프레임에서 탐지됨")
print(f"[💾 저장된 영상] {output_path}")
```

## 📊 모델 성능

### 전체 성능 지표
- **mAP@50**: 0.92
- **mAP@50-95**: 0.78

### 클래스별 성능

| Class Name | Precision | Recall | mAP@50 | mAP@50-95 |
|------------|-----------|--------|---------|-----------|
| tire | 0.975 | 0.989 | 0.994 | 0.967 |
| spring fish trap | 0.972 | 0.978 | 0.978 | 0.951 |
| circular fish trap | 0.955 | 0.969 | 0.985 | 0.962 |
| eel fish trap | 0.962 | 0.977 | 0.992 | 0.926 |
| rectangular fish trap | 0.989 | 0.960 | 0.991 | 0.921 |
| fish net | 0.922 | 0.895 | 0.930 | 0.738 |
| wood | 0.891 | 0.934 | 0.958 | 0.851 |
| bundle of ropes | 0.760 | 0.716 | 0.777 | 0.578 |
| rope | 0.675 | 0.506 | 0.595 | 0.428 |
| other objects | 1.0 | 0.0 | 0.995 | 0.497 |

### 📌 분석
**✅ 강점**
- 대부분 클래스가 mAP@50 > 0.9 → 상업용 영상 탐지도 가능
- tire, trap 계열 클래스는 정밀도와 재현율 모두 95% 이상

**⚠️ 개선 필요 클래스**
- rope: 객체 경계가 모호하거나 작은 형태로 인해 성능 저조
- other objects: 라벨이 1개뿐 → 리콜 0은 당연함 (미탐일 가능성)

## 🛠️ 유틸리티 스크립트

### XML to YOLO 변환 스크립트
```python
import os
import xml.etree.ElementTree as ET

ANNOTATION_DIR = './annotations'
LABEL_DIR = './labels_yolo'
os.makedirs(LABEL_DIR, exist_ok=True)

class_map = {
    "tire": 0,
    "wood": 1,
    "rope": 2,
    "spring fish trap": 3,
    "bundle of ropes": 4,
    "circular fish trap": 5,
    "eel fish trap": 6,
    "fish net": 7,
    "rectangular fish trap": 8,
    "other objects": 9
}

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x_center * dw, y_center * dh, w * dw, h * dh)

for filename in os.listdir(ANNOTATION_DIR):
    if not filename.endswith('.xml'):
        continue

    tree = ET.parse(os.path.join(ANNOTATION_DIR, filename))
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    out_path = os.path.join(LABEL_DIR, filename.replace('.xml', '.txt'))
    with open(out_path, 'w') as out_file:
        for obj in root.findall('object'):
            class_name = obj.find('name').text.strip().lower()
            if class_name not in class_map:
                continue  # 무시할 클래스
            cls_id = class_map[class_name]
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            bbox = convert_bbox((w, h), (xmin, xmax, ymin, ymax))
            out_file.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bbox])}\n")
```

### 데이터셋 분할 스크립트
```python
import os
import random
import shutil

# 경로 설정
IMG_DIR = '/home/YOLO/images'
LABEL_DIR = '/home/YOLO/labels_yolo'
DEST = '/home/YOLO/dataset'

# 결과 폴더 생성
for phase in ['train', 'val']:
    os.makedirs(f"{DEST}/images/{phase}", exist_ok=True)
    os.makedirs(f"{DEST}/labels/{phase}", exist_ok=True)

# 이미지 파일 중 라벨이 있는 것만 추출
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
image_files_with_labels = []
for f in image_files:
    name = os.path.splitext(f)[0]
    label_path = os.path.join(LABEL_DIR, name + '.txt')
    if os.path.exists(label_path):
        image_files_with_labels.append(f)
    else:
        print(f"[SKIP] No label for {f}")

# 랜덤 셔플 후 split
random.shuffle(image_files_with_labels)
split_ratio = 0.9
split_index = int(len(image_files_with_labels) * split_ratio)
train_files = image_files_with_labels[:split_index]
val_files = image_files_with_labels[split_index:]

# 파일 복사
for phase, files in [('train', train_files), ('val', val_files)]:
    for f in files:
        name = os.path.splitext(f)[0]
        shutil.copy(os.path.join(IMG_DIR, f), f"{DEST}/images/{phase}/{f}")
        shutil.copy(os.path.join(LABEL_DIR, name + '.txt'), f"{DEST}/labels/{phase}/{name}.txt")

print(f"[DONE] {len(train_files)} train, {len(val_files)} val 이미지가 준비되었습니다.")
```

### 클래스 분포 확인 스크립트
```python
import os
import xml.etree.ElementTree as ET
from collections import Counter

ANNOTATION_DIR = '/home/work/.exdata98/test/YOLO/annotations'
class_counter = Counter()

for filename in os.listdir(ANNOTATION_DIR):
    if not filename.endswith('.xml'):
        continue

    xml_path = os.path.join(ANNOTATION_DIR, filename)
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            name = obj.find('name').text.strip().lower()
            class_counter[name] += 1

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")

# 결과 출력
print("\n🧾 추출된 고유 클래스 목록:")
for i, (cls, count) in enumerate(class_counter.most_common()):
    print(f"{i:2d}: {cls} ({count}개)")
```

##
