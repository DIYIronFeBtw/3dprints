import requests
import torch
from PIL import Image, ImageDraw
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline

# Prompt for the AI to know what to generate in the void
prompt = "subtle cave high resolution"

# Set crop width to 40% of 512
crop_width = int(512 * 0.4)
crop_offset = 100

input_image = Image.open("photo1.jpg")

# Scale everything for 512x512
scale_factor = 512 / input_image.height
input_image = input_image.resize((int(input_image.width * scale_factor), 512))

# Generate input image
left_crop = input_image.crop((0, 0, crop_width, input_image.height))
right_crop = input_image.crop((input_image.width - crop_width, 0, input_image.width, input_image.height))

# Create a new image to feed into StableDiffusion
sd_input_image = Image.new("RGB", (512, 512))
sd_input_image.paste(right_crop, (0, 0))
sd_input_image.paste(left_crop, (sd_input_image.width - crop_width, 0))

# Generate Mask
mask = Image.new("RGB", (sd_input_image.width, sd_input_image.height))
draw = ImageDraw.Draw(mask)
draw.rectangle(
    ((crop_width - crop_offset, 0), (sd_input_image.width - crop_width + crop_offset, sd_input_image.height)),
    fill="white")

inner_image = sd_input_image.convert("RGBA")

model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    variant='fp16',
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe = pipe.to("cuda")

images = pipe(prompt=prompt,
              image=sd_input_image,
              num_inference_steps=60,  # higher for more quality but longer calculation
              guidance_scale=6,  # Higher for stricter/sharper, lower for creative
              mask_image=mask).images

# The output image
ai_output_image = images[0]
ai_output_image.save("aiOutput.jpg")

# Cut the output in half
right_crop = ai_output_image.crop((0, 0, int(ai_output_image.width / 2), ai_output_image.height))
left_crop = ai_output_image.crop((int(ai_output_image.width / 2), 0, ai_output_image.width, ai_output_image.height))

# Crop the original image to only have middle
input_image = input_image.crop((crop_width, 0, input_image.width - crop_width, input_image.height))

# Generate final output image
output_image = Image.new("RGB", (input_image.width + ai_output_image.width, input_image.height))

# Paste left, middle, right
output_image.paste(left_crop, (0, 0))
output_image.paste(input_image, (left_crop.width, 0))
output_image.paste(right_crop, (left_crop.width + input_image.width, 0))

# Save the actual output
output_image.save("output.jpg")

# Create another image to show looping demo
output_image2 = Image.new("RGB", (output_image.width * 3, output_image.height))
output_image2.paste(output_image, (0, 0))
output_image2.paste(output_image, (output_image.width, 0))
output_image2.paste(output_image, (output_image.width * 2, 0))
output_image2.save("effect-demo.jpg")
