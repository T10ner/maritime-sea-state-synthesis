from PIL import Image
import os
import numpy as np

input_mask_folder = r"F:\mart_proj\masks"         # 你已有的船mask：白船黑底
input_image_folder = r"F:\mart_proj\frames"       # 原图
output_folder = r"F:\mart_proj\inpaint_masks"     # 最终给inpainting用的mask

os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_image_folder) if f.endswith(".jpg")]

for img_file in image_files:
    image_path = os.path.join(input_image_folder, img_file)
    mask_path = os.path.join(input_mask_folder, img_file)

    if not os.path.exists(mask_path):
        continue

    image = Image.open(image_path).convert("RGB")
    ship_mask = Image.open(mask_path).convert("L")

    w, h = image.size
    ship_mask_np = np.array(ship_mask)

    # 先做一个全黑mask（默认全部保护）
    final_mask = np.zeros((h, w), dtype=np.uint8)

    # 只开放“海面中间区域”给模型编辑
    # 上面35%不动（天空）
    # 最下面5%也先不动（避免近景乱纹理）
    # 地平线附近留保护带
    y_top = int(h * 0.45)
    y_bottom = int(h * 0.92)

    final_mask[y_top:y_bottom, :] = 255

    # 再把船保护回来：船所在区域设为0
    final_mask[ship_mask_np > 0] = 0

    # 地平线附近额外保护一条带，避免地平线变形
    horizon_band_top = int(h * 0.43)
    horizon_band_bottom = int(h * 0.50)
    final_mask[horizon_band_top:horizon_band_bottom, :] = 0

    out_path = os.path.join(output_folder, img_file)
    Image.fromarray(final_mask).save(out_path)

print("inpaint mask 生成完成")
