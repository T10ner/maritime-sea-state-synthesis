from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageOps
import torch
import os

# ===== 你要改的两行 =====
image_path = r"F:\mart_proj\frames\MVI_1622_VIS_frame_0094.jpg"
mask_path = r"F:\mart_proj\masks\MVI_1622_VIS_frame_0094.jpg"

# ===== 输出目录 =====
output_dir = r"F:\mart_proj\outputs"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "demo_result.png")
invert_mask_path = os.path.join(output_dir, "demo_mask_inverted.png")

# ===== 读取原图和mask =====
image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

# 反相mask：
# 原mask 白=船 黑=海面
# inpainting常用 白=要重绘 黑=保留
mask = ImageOps.invert(mask)
mask.save(invert_mask_path)

# ===== 加载模型 =====
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 可选：节省显存
pipe.enable_attention_slicing()

# ===== Prompt =====
prompt = (
    "realistic harbor water, sea state 3, moderate waves, small whitecaps, natural wave pattern, gentle motion, not stormy "
    "natural maritime scene, realistic ocean surface, windy but not stormy, high detail"
)

negative_prompt = (
    "distorted ship, broken ship, unrealistic vessel, extra boats, blurry, artifacts, "
    "deformed horizon, edited sky, unrealistic lighting, warped dock, storm, big waves, splash, spray, huge waves"
)

# ===== 生成 =====
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    mask_image=mask,
    guidance_scale=7.5,
    num_inference_steps=50,
    strength=0.75
).images[0]

result.save(output_path)
print("生成完成：", output_path)
print("反相mask已保存：", invert_mask_path)
