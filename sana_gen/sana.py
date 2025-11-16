# test sana sprint
from diffusers import SanaSprintPipeline
import torch

pipeline = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16
)

pipeline.to("cuda:0")

prompt = "a tiny astronaut hatching from an egg on the moon"

image = pipeline(prompt=prompt, num_inference_steps=2).images[0]
image.save("test_out.png")