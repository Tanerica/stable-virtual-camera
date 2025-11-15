import argparse
import functools
import random
import json  
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig


def process_prompt_data(index, save_paths, prompt_embed, output_path):
    np.save(output_path / save_paths[index], prompt_embed)
    return index


def wrapper_process_prompt_data(args):
    return process_prompt_data(*args)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def encode_prompt(prompts, text_encoder, tokenizer, is_train=True):
    captions = []
    for caption in prompts:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
        )[0]

    return {"prompt_embeds": prompt_embeds.detach().cpu().numpy()}


def main(args):
    batch_size = args.batch_size
    
    # 🔧 MODIFIED: Read JSON prompt file
    with open(args.prompt_list) as f:
        data = json.load(f)

    prompts = [item["prompt"] for item in data]
    save_paths = [Path(item["path"]).with_suffix(".pt").name for item in data]
   
    op = Path(args.out_dir)
    op.mkdir(exist_ok=True, parents=True)
    
    from diffusers import SanaSprintPipeline
    from torchvision import transforms
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("lzq49/mvdream-sd21-diffusers", subfolder="vae")
    vae = vae.to("cuda")
    pipe = SanaSprintPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    preprocess = transforms.Compose([
            transforms.Resize((512, 512)),   # Resize to VAE input size
            transforms.ToTensor(),           # Convert to tensor (C, H, W)
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
    from tqdm.auto import tqdm
    pipe.text_encoder = torch.compile(pipe.text_encoder)
    pipe.transformer = torch.compile(pipe.transformer)
    pipe.vae = torch.compile(pipe.vae)
    vae.encode = torch.compile(vae.encode)
    
    import time
    start = True
    for prompt, save_path in tqdm(zip(prompts, save_paths), total=len(prompts), desc="Processing prompts"):
        image = pipe(prompt=prompt, num_inference_steps=2, height=512, width=512, num_images_per_prompt=4).images
        
        init_image_tensor = torch.stack([preprocess(img) for img in image]).to("cuda")

        with torch.no_grad():
            latent_dist = vae.encode(init_image_tensor).latent_dist
            latent = latent_dist.sample()
            latent = latent * 0.18215  # Scaling factor used by Stable Diffusion
        # from PIL import Image
        # for i, img_array in enumerate(decoded):
        #     img = Image.fromarray((img_array * 255).astype("uint8"))
        #     img.save(f"sana_test_{i}.png")
        torch.save(latent, op / save_path)
        if start:
           start_time = time.time()
           start = False
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt encoder with JSON input.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out",
        required=True,
        help="Directory to store output .npy embeddings.",
    )
    parser.add_argument(
        "--prompt_list",
        type=str,
        default=None,
        required=True,
        help="Path to a JSON file with `prompt` and `path` keys.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for encoding the text embedding.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=16,
        help="Number of processes for writing embeddings.",
    )

    args = parser.parse_args()
    main(args)


