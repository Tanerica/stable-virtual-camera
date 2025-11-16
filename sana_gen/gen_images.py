import os
from diffusers import SanaSprintPipeline
import torch

# read_prompts.py
prompts = []
with open("caption100.txt", "r", encoding="utf-8") as f:
    # strip removes newline characters (\n)
    prompts = [line.strip() for line in f.readlines() if line.strip()]

output_dir = './data'
os.makedirs(output_dir, exist_ok=True)
pipeline = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16
).to('cuda')

#pipeline.transformer = torch.compile(pipeline.transformer)
#pipeline.text_encoder = torch.compile(pipeline.text_encoder)
#pipeline.vae = torch.compile(pipeline.vae)


for idx, prompt in enumerate(prompts):
    image = pipeline(prompt="A 3D small centric object of " + prompt, num_inference_steps=2, height=576, width=576).images[0]
    filename = f'{idx:06d}.png'
    save_path = os.path.join(output_dir, filename)
    image.save(save_path)

# image = pipeline(prompt=prompt, num_inference_steps=2).images[0]
# image.save("test_out.png")