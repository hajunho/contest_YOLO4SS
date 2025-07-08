import os
import xml.etree.ElementTree as ET

# 경로 설정
ANNOTATIONS_DIR = './annotations'
LABELS_DIR = './labels_yolo'
os.makedirs(LABELS_DIR, exist_ok=True)

# 클래스 사전 (하나뿐)
class_map = {"bundle of ropes": 0}

def convert_bbox(size, box):
    dw = 1. / size[0]  # width
    dh = 1. / size[1]  # height
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x_center * dw, y_center * dh, w * dw, h * dh)

for file in os.listdir(ANNOTATIONS_DIR):
    if not file.endswith('.xml'):
        continue

    in_file = open(os.path.join(ANNOTATIONS_DIR, file))
    tree = ET.parse(in_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    label_filename = os.path.join(LABELS_DIR, file.replace('.xml', '.txt'))
    with open(label_filename, 'w') as out_file:
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in class_map:
                continue
            cls_id = class_map[cls_name]

            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymin = float(xmlbox.find('ymin').text)
            ymax = float(xmlbox.find('ymax').text)

            # YOLO 형식으로 변환
            bb = convert_bbox((width, height), (xmin, xmax, ymin, ymax))
            out_file.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")

