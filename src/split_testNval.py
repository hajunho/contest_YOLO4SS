import os
import random
import shutil

# 경로 설정
IMG_DIR = '/home/work/.exdata98/test/YOLO/images'
LABEL_DIR = '/home/work/.exdata98/test/YOLO/labels_yolo'
DEST = '/home/work/.exdata98/test/YOLO/dataset'

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

