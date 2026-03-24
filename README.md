港口场景三级海况数据合成与图像处理方法研究
一、项目背景
本项目旨在针对港口泊区场景，构建一套基于真实视频数据的海况图像合成流程，用于支持后续船舶检测、异常行为分析等任务。
由于真实环境中：

不同海况（尤其三级海况）数据较难获取；
标注成本较高；
场景复杂（船舶密集、光照变化大）。

因此，本项目探索一种“检测 + 去除 + 合成”的数据生成方案。

二、总体思路
本项目核心思路为：
从真实港口视频中提取图像 → 检测并去除船舶 → 得到“纯海面背景” → 用于后续海况合成
流程如下：

视频抽帧（构建数据基础）
船舶检测（YOLOv8）
目标区域Mask生成
图像修复（Inpainting）去除船舶
得到干净海面数据


三、方法设计
1. 数据来源

数据集：Singapore Maritime Dataset（SMD）
数据类型：岸基摄像头拍摄的港口视频
内容：包含大量船舶、浮标、不同天气与光照条件


2. 视频抽帧
使用 OpenCV 对视频进行抽帧：

设置固定帧间隔（如每5帧或每5秒）
降低冗余帧，避免连续相似画面

输出文件夹：
textframes/
├── train/
└── test/

3. 船舶检测（YOLOv8）
使用 Ultralytics YOLOv8 进行目标检测：

模型：yolov8n.pt（预训练模型）
检测类别：boat（船舶）

特点：

推理速度快
无需额外训练即可使用

输出文件夹：
textdetections/

4. Mask生成
根据检测框生成二值Mask：

白色区域（255）：船舶（需要去除）
黑色区域（0）：背景

输出文件夹：
textmasks/

5. Inpainting（图像修复）
使用 Hugging Face Diffusers 中的扩散模型：

模型：Stable Diffusion Inpainting
输入：原图 + Mask
输出：去除船舶后的干净海面图像

输出文件夹：
textoutputs/
├── outputs_batch/

四、实验结果
1. 成功效果

能有效去除船舶目标
保留海面纹理与颜色一致性
可生成“无船舶干净海面”

2. 存在问题

小目标（远处船舶）存在漏检情况
浮标可能被误识别为船舶
Inpainting区域在个别情况下存在轻微不自然现象


五、关键技术点

目标检测：YOLOv8（Ultralytics最新版）
图像处理：OpenCV
生成模型：Diffusion（Stable Diffusion）
GPU加速：PyTorch + CUDA


六、项目结构
textmart_proj/
├── extract_frames.py          # 视频抽帧
├── detect_ships.py            # 船舶检测
├── generate_mask.py           # Mask生成
├── generate_inpaint_mask.py   # Inpainting Mask处理
├── inpaint_demo.py            # 单张测试
├── inpaint_batch.py           # 批量处理
└── README.md
