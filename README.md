# 港口场景三级海况数据合成与图像处理方法研究


## 一、总体思路

本项目核心思路为：

从真实港口视频中提取图像 → 检测并船舶 → 反向得到纯海面背景 → 生成三级海况

流程如下：

1. 视频抽帧（构建数据基础）
2. 船舶检测（YOLOv8）
3. 目标区域Mask生成
4. 图像修复（Inpainting）去除船舶
5. 得到干净海面数据

---

## 二、方法设计

### 1. 数据来源

- 数据集：Singapore Maritime Dataset（SMD）
- 数据类型：岸基摄像头拍摄的港口视频
- 内容：包含大量船舶、浮标、不同天气与光照条件

---

### 2. 视频抽帧

使用 OpenCV 对视频进行抽帧：

- 设置时间间隔（如每5秒）
- 降低冗余帧（避免连续相似画面）

输出目录：
frames/
![MVI_1474_VIS_frame_0008](https://github.com/user-attachments/assets/ddd85343-fa35-4574-9b5f-14c6565c6e49)
![MVI_1478_VIS_frame_0011](https://github.com/user-attachments/assets/be07ed85-652e-4ef4-8874-b802df7f2543)


---

### 3. 船舶检测（YOLOv8）

使用 Ultralytics YOLOv8（最新版）进行目标检测：

- 模型：yolov8n.pt（预训练模型）
- 检测类别：boat（船舶）



输出目录：
detections/
![MVI_1474_VIS_frame_0009](https://github.com/user-attachments/assets/ec0ec979-aaea-434a-822d-972de0cd9b95)
![MVI_1478_VIS_frame_0014](https://github.com/user-attachments/assets/5b9b3402-c8c9-4908-87f7-270dfda9df95)

---

### 4. Mask生成

根据检测框生成二值Mask：

- 白色区域：船舶（需要去除）
- 黑色区域：背景

输出目录：
masks/
![MVI_1474_VIS_frame_0008](https://github.com/user-attachments/assets/42ebe139-a438-4292-9dbb-a068902c0de3)
![MVI_1478_VIS_frame_0013](https://github.com/user-attachments/assets/4a3fbd7f-df1c-4d47-8854-d4c27de9e3d4)

---

### 5. 图像生成

使用 HuggingFace Diffusers 中的扩散模型：

- 模型：Stable Diffusion Inpainting
- 输入：原图 + mask
- 输出：三级海况下与原图船舶的合成

输出目录：
outputs/
outputs_batch/

![MVI_1474_VIS_frame_0008](https://github.com/user-attachments/assets/322aff53-4537-4c42-8e3e-6fe1499228bb)
![MVI_1478_VIS_frame_0013](https://github.com/user-attachments/assets/a3b2ea87-6168-4ea4-a0d3-48e275f8678c)

---

## 三、实验结果

### 1. 成功效果

- 能有效去除船舶反向目标
- 保留海面纹理与颜色一致性
- 可生成“干净三级海况下与原图船舶的合成”
  
![MVI_1474_VIS_frame_0008](https://github.com/user-attachments/assets/322aff53-4537-4c42-8e3e-6fe1499228bb)
![MVI_1478_VIS_frame_0013](https://github.com/user-attachments/assets/a3b2ea87-6168-4ea4-a0d3-48e275f8678c)


### 2. 存在问题

- 小目标（远处船）存在漏检
- 浮标可能被误识别为船
- 局部区域偶尔不自然

![MVI_1469_VIS_frame_0002](https://github.com/user-attachments/assets/cfbe0033-b519-4d60-896d-f824f6189a1d)
![MVI_1486_VIS_frame_0036](https://github.com/user-attachments/assets/c2345ed3-be0a-42a1-b9f3-3e633671e1bf)
![MVI_1469_VIS_frame_0001](https://github.com/user-attachments/assets/fadbee66-63f4-4b02-8f68-10ecae473861)


---

## 四、关键技术点

- 目标检测：YOLOv8
- 图像处理：OpenCV
- 生成模型：Diffusion（Stable Diffusion）
- GPU加速：PyTorch + CUDA

---

## 五、项目结构

mart_proj/
├── extract_frames.py        # 视频抽帧
├── detect_ships.py         # 船舶检测
├── generate_mask.py        # mask生成
├── generate_inpaint_mask.py# inpainting mask处理
├── inpaint_demo.py         # 单张测试
├── inpaint_batch.py        # 批量处理

---

