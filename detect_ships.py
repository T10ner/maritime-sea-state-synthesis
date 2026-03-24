from ultralytics import YOLO
import cv2
import os

# 路径
image_folder = r"F:\mart_proj\frames"
output_folder = r"F:\mart_proj\detections"

os.makedirs(output_folder, exist_ok=True)

# 加载模型
model = YOLO("yolov8n.pt")

image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

print(f"检测 {len(image_files)} 张图片...")

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)

    # 🔍检测
    results = model(img_path)[0]

    # 读取原图
    img = cv2.imread(img_path)

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # COCO里：船 = class 8（boat）
        if cls == 8 and conf > 0.3:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 画框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 写标签
            cv2.putText(img, f"boat {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 保存结果
    save_path = os.path.join(output_folder, img_file)
    cv2.imwrite(save_path, img)

print("检测完成！")
