import cv2
from ultralytics import YOLO

model_path = './best.pt'
video_path = './seetrash_val_02.mp4'
output_path = './seetrash_val_02_detected.mp4'

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

