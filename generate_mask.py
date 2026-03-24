from ultralytics import YOLO
import cv2
import os
import numpy as np

image_folder = r"F:\mart_proj\frames"
mask_folder = r"F:\mart_proj\masks"

os.makedirs(mask_folder, exist_ok=True)

model = YOLO("yolov8n.pt")

image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)

    results = model(img_path)[0]
    img = cv2.imread(img_path)

    h, w = img.shape[:2]

    # 黑色mask（默认全是0）
    mask = np.zeros((h, w), dtype=np.uint8)

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 8 and conf > 0.2:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ⭐扩展框（关键！）
            box_w = x2 - x1
            box_h = y2 - y1
            expand = max(12, int(min(box_w, box_h) * 0.35))
            x1 = max(0, x1 - expand)
            y1 = max(0, y1 - expand)
            x2 = min(w, x2 + expand)
            y2 = min(h, y2 + expand)

            # 填充为白色（255）
            mask[y1:y2, x1:x2] = 255

    save_path = os.path.join(mask_folder, img_file)
    cv2.imwrite(save_path, mask)

print("mask生成完成！")
