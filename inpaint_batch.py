from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageOps
import torch
import os

frames_folder = r"F:\mart_proj\frames"
masks_folder = r"F:\mart_proj\masks"
output_folder = r"F:\mart_proj\outputs_batch"

os.makedirs(output_folder, exist_ok=True)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_attention_slicing()

prompt = (
    "realistic harbor water, sea state 3, moderate waves, small whitecaps, natural wave pattern, gentle motion, not stormy "
    "natural maritime scene, realistic ocean surface, windy but not stormy, high detail"
)

negative_prompt = (
    "distorted ship, broken ship, unrealistic vessel, extra boats, blurry, artifacts, "
    "deformed horizon, edited sky, unrealistic lighting, warped dock, storm, big waves, splash, spray, huge waves"
)

image_files = [f for f in os.listdir(frames_folder) if f.endswith(".jpg")]

# 只跑前15张（防止太慢）
for img_file in image_files[:15]:
    image_path = os.path.join(frames_folder, img_file)
    mask_path = os.path.join(masks_folder, img_file)

    if not os.path.exists(mask_path):
        continue

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    mask = ImageOps.invert(mask)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        guidance_scale=7.5,
        num_inference_steps=50,
        strength=0.75
    ).images[0]

    save_path = os.path.join(output_folder, img_file)
    result.save(save_path)

    print("完成:", img_file)

print("全部完成")
