import os.path as osp
from datetime import datetime
import os
import imageio.v3 as iio
import numpy as np
import torch
import tyro
from einops import repeat
from seva.eval import transform_img_and_K, get_value_dict, save_output, seed_everything, decode_output
from seva.geometry import DEFAULT_FOV_RAD, get_default_intrinsics, get_preset_pose_fov
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DiscreteDenoiser
from seva.utils import load_model
from seva.sampling import (
    EulerEDMSampler,
    MultiviewCFG,
)

device = "cuda:0"

# Constants
WORK_DIR = "work_dirs/demo_cli"
model = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
ae = AutoEncoder(chunk_size=1).to(device)
conditioner = CLIPConditioner().to(device)
denoiser = DiscreteDenoiser(num_idx=1000, device=device)
T = 21
num_steps = 50
cfg_min = 1.2


def preprocess(input_img_path: str):
    shorter = 576  # must divided by 64

    input_imgs = torch.as_tensor(
        iio.imread(input_img_path) / 255.0, dtype=torch.float32
    )[None, ..., :3]
    input_imgs = transform_img_and_K(
        input_imgs.permute(0, 3, 1, 2), shorter, K=None, size_stride=64
    )[0].permute(0, 2, 3, 1)

    input_Ks = get_default_intrinsics(
        aspect_ratio=input_imgs.shape[2] / input_imgs.shape[1]
    )
    input_c2ws = torch.eye(4)[None]

    return {
        "input_imgs": input_imgs,
        "input_Ks": input_Ks,
        "input_c2ws": input_c2ws,
        "input_wh": (input_imgs.shape[2], input_imgs.shape[1]),
    }


def get_target_c2ws_and_Ks_from_preset(
    preprocessed: dict, preset_traj: str, num_frames: int, zoom_factor: float
):
    img_wh = preprocessed["input_wh"]
    start_c2w = preprocessed["input_c2ws"][0]
    start_w2c = torch.linalg.inv(start_c2w)
    look_at = torch.tensor([0, 0, 10])
    start_fov = DEFAULT_FOV_RAD

    target_c2ws, target_fovs = get_preset_pose_fov(
        preset_traj,
        num_frames,
        start_w2c,
        look_at,
        -start_c2w[:3, 1],
        start_fov,
        spiral_radii=[1.0, 1.0, 0.5],
        zoom_factor=zoom_factor,
    )
    target_c2ws = torch.as_tensor(target_c2ws)
    target_fovs = torch.as_tensor(target_fovs)
    target_Ks = get_default_intrinsics(
        target_fovs, aspect_ratio=img_wh[0] / img_wh[1]
    )
    return target_c2ws, target_Ks


def main(
    input_img: str,
    seed: int = 23,
    cfg: float = 4.0,
    preset_traj: str = "orbit",
    num_frames: int = 20,
    zoom_factor: float = 1.0,
    camera_scale: float = 2.0,
):
    torch.cuda.reset_peak_memory_stats()
    preprocessed = preprocess(input_img)

    render_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = osp.join(WORK_DIR, render_name)

    input_imgs, input_Ks, input_c2ws, (W, H) = (
        preprocessed["input_imgs"],
        preprocessed["input_Ks"],
        preprocessed["input_c2ws"],
        preprocessed["input_wh"],
    )

    num_inputs = len(input_imgs)
    input_c2ws = input_c2ws.to(dtype=torch.float32)
    target_c2ws, target_Ks = get_target_c2ws_and_Ks_from_preset(
        preprocessed, preset_traj, num_frames, zoom_factor
    )
    all_c2ws = torch.cat([input_c2ws, target_c2ws], 0)
    all_Ks = torch.cat([input_Ks, target_Ks], 0) * input_Ks.new_tensor([W, H, 1])[:, None]
    num_targets = len(target_c2ws)
    input_indices = list(range(num_inputs))

    traj_prior_c2ws = all_c2ws[1:]

    # Conditioning (torch-only, no numpy roundtrip)
    image_cond = {
        "input_indices": input_indices,
    }
    camera_cond = {
        "c2w": all_c2ws,
        "K": all_Ks,
        "input_indices": list(range(num_inputs + num_targets)),
    }

    # Options
    print("Rendering started...")

    # --- Normalize input images only ---
    input_imgs = input_imgs[0].permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    input_imgs = input_imgs * 2.0 - 1.0  # already in [0,1], scale to [-1,1]

    # --- Get corresponding poses ---
    input_c2ws = camera_cond["c2w"][image_cond["input_indices"]]
    input_Ks = camera_cond["K"][image_cond["input_indices"]]

    # --- Save inputs ---
    print(f"Saving input images to {save_path} ...")
    save_output(
        {"/image": input_imgs},
        save_path=os.path.join(save_path, "input"),
        video_save_fps=2,
    )

    torch.cuda.empty_cache()
    seed_everything(seed)

    # --- Simplified case: directly concat input + priors ---
    print("simplified 1-input + N-tests case")

    curr_input_sels = [0]  # input always at index 0
    curr_imgs = torch.cat(
        [input_imgs, input_imgs.new_zeros(len(traj_prior_c2ws), *input_imgs.shape[1:])],
        dim=0,
    )
    curr_c2ws = torch.cat([input_c2ws, traj_prior_c2ws], dim=0)
    curr_Ks = torch.cat([input_Ks, all_Ks[1:]], dim=0)

    # --- Build value_dict ---
    value_dict = get_value_dict(
        curr_imgs.to("cuda"),
        curr_input_sels,
        curr_c2ws,
        curr_Ks,
        list(range(T)),
        all_c2ws=camera_cond["c2w"],
        camera_scale=camera_scale,
    )

    sampler = EulerEDMSampler(
        discretization=denoiser.discretization,
        guider=MultiviewCFG(cfg_min),
        num_steps=num_steps,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=999.0,
        s_noise=1.0,
        verbose=True,
        device=device,
    )

    imgs = value_dict["cond_frames"].to("cuda")
    input_masks = value_dict["cond_frames_mask"].to("cuda")
    pluckers = value_dict["plucker_coordinate"].to("cuda")

    # --- Encode ---
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        latents = torch.nn.functional.pad(
            ae.encode(imgs[input_masks], 1), (0, 0, 0, 0, 0, 1), value=1.0
        )

        c_crossattn = repeat(conditioner(imgs[input_masks]).mean(0), "d -> n 1 d", n=T)
        uc_crossattn = torch.zeros_like(c_crossattn)

        c_replace = latents.new_zeros(T, *latents.shape[1:])
        c_replace[input_masks] = latents
        uc_replace = torch.zeros_like(c_replace)

        c_concat = torch.cat(
            [
                repeat(
                    input_masks,
                    "n -> n 1 h w",
                    h=pluckers.shape[2],
                    w=pluckers.shape[3],
                ),
                pluckers,
            ],
            1,
        )
        uc_concat = torch.cat(
            [pluckers.new_zeros(T, 1, *pluckers.shape[-2:]), pluckers], 1
        )

        c_dense_vector = pluckers
        uc_dense_vector = c_dense_vector

        c = {
            "crossattn": c_crossattn,
            "replace": c_replace,
            "concat": c_concat,
            "dense_vector": c_dense_vector,
        }
        uc = {
            "crossattn": uc_crossattn,
            "replace": uc_replace,
            "concat": uc_concat,
            "dense_vector": uc_dense_vector,
        }

        additional_model_inputs = {"num_frames": T}
        additional_sampler_inputs = {
            "c2w": value_dict["c2w"].to("cuda"),
            "K": value_dict["K"].to("cuda"),
            "input_frame_mask": value_dict["cond_frames_mask"].to("cuda"),
        }

        randn = torch.randn((T, 4, H // 8, W // 8)).to("cuda")

        samples_z = sampler(
            lambda input, sigma, c: denoiser(
                model,
                input,
                sigma,
                c,
                **additional_model_inputs,
            ),
            randn,
            scale=float(cfg),
            cond=c,
            uc=uc,
            verbose=True,
            **additional_sampler_inputs,
        )
        samples = ae.decode(samples_z, 1)

        samples = decode_output(samples[1:], T)

        # Save outputs
        save_output(
            samples,
            save_path=os.path.join(save_path, "first-pass"),
            video_save_fps=5,
        )
        video_path_0 = os.path.join(save_path, "first-pass", "samples-rgb.mp4")
        print(f"Generated video : {video_path_0}")
    print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Peak memory reserved:  {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    print(torch.cuda.mem_get_info())
    free, total = torch.cuda.mem_get_info()
    print(f"Free: {free/1024**3:.2f} GB, Total: {total/1024**3:.2f} GB")


if __name__ == "__main__":
    tyro.cli(main)
