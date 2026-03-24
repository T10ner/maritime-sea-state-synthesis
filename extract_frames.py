import cv2
import os

video_folder = r"F:\mart_proj\videos"
output_folder = r"F:\mart_proj\frames"

os.makedirs(output_folder, exist_ok=True)

video_files = [
    f for f in os.listdir(video_folder)
    if f.endswith(".avi") and "Haze" not in f
]

interval_seconds = 5 #5秒取一帧
MAX_IMAGES = 100  # 总数

saved_count = 0

for video_file in video_files:
    if saved_count >= MAX_IMAGES:
        break

    video_path = os.path.join(video_folder, video_file)
    print(f"处理视频: {video_file}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开: {video_file}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = max(1, int(fps * interval_seconds))

    frame_count = 0

    while True:
        if saved_count >= MAX_IMAGES:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval_frames == 0:
            filename = f"{os.path.splitext(video_file)[0]}_frame_{saved_count:04d}.jpg"
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

print(f"完成，共保存 {saved_count} 张图像")
