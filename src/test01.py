import cv2
from ultralytics import YOLO

# 현재 폴더 기준 상대 경로
model_path = './best.pt'
video_path = './seetrash_val_01.mp4'
output_path = './seetrash_val_01_detected.mp4'

# 모델 로드
model = YOLO(model_path)

# 비디오 열기
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 출력 비디오 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 프레임별로 탐지 및 시각화
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
print(f"[완료] 시각화 영상 저장됨 → {output_path}")

