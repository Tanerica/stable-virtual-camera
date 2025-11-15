import os.path as osp
from datetime import datetime
import imageio.v3 as iio
import numpy as np
import torch
from einops import repeat
import tyro

from seva.eval import (
    transform_img_and_K,
    get_value_dict,
    save_output,
    seed_everything,
    decode_output,
)
from seva.geometry import DEFAULT_FOV_RAD, get_default_intrinsics, get_preset_pose_fov
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DiscreteDenoiser, EulerEDMSampler, MultiviewCFG
from seva.utils import load_model

# --- Config ---
device = "cuda:0"
WORK_DIR = "work_dirs/demo_cli"
T = 21

# --- Load models ---
model = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
ae = AutoEncoder(chunk_size=1).to(device)
conditioner = CLIPConditioner().to(device)
denoiser = DiscreteDenoiser(num_idx=1000, device=device)


def preprocess(input_img_path: str):
    shorter = round(576 / 64) * 64  # ensure multiple of 64

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


class SevaRenderer:
    def get_target_c2ws_and_Ks_from_preset(
        self, preprocessed: dict, preset_traj: str, num_frames: int, zoom_factor: float
    ):
        W, H = preprocessed["input_wh"]
        start_c2w = preprocessed["input_c2ws"][0]
        start_w2c = torch.linalg.inv(start_c2w)

        target_c2ws, target_fovs = get_preset_pose_fov(
            preset_traj,
            num_frames,
            start_w2c,
            look_at=torch.tensor([0, 0, 10]),
            up=-start_c2w[:3, 1],
            start_fov=DEFAULT_FOV_RAD,
            spiral_radii=[1.0, 1.0, 0.5],
            zoom_factor=zoom_factor,
        )
        target_c2ws = torch.as_tensor(target_c2ws)
        target_Ks = get_default_intrinsics(target_fovs, aspect_ratio=W / H)
        return target_c2ws, target_Ks

    def render(
        self,
        preprocessed: dict,
        seed: int,
        cfg: float,
        preset_traj: str,
        num_frames: int,
        zoom_factor: float,
        camera_scale: float,
    ):
        render_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = osp.join(WORK_DIR, render_name)

        input_imgs, input_Ks, input_c2ws, (W, H) = (
            preprocessed["input_imgs"],
            preprocessed["input_Ks"],
            preprocessed["input_c2ws"],
            preprocessed["input_wh"],
        )

        # --- Trajectory setup ---
        target_c2ws, target_Ks = self.get_target_c2ws_and_Ks_from_preset(
            preprocessed, preset_traj, num_frames, zoom_factor
        )
        all_c2ws = torch.cat([input_c2ws, target_c2ws], 0)
        all_Ks = torch.cat([input_Ks, target_Ks], 0) * input_Ks.new_tensor([W, H, 1])[:, None]

        num_inputs = len(input_imgs)
        num_targets = len(target_c2ws)

        # Anchor selection
        anchor_indices = np.linspace(num_inputs, num_inputs + num_targets - 1, T - num_inputs)
        traj_prior_c2ws = all_c2ws[anchor_indices.round().long()]
        traj_prior_Ks = all_Ks[anchor_indices.round().long()]

        # Conditioning
        all_imgs_np = (
            torch.nn.functional.pad(input_imgs, (0, 0, 0, 0, 0, 0, 0, num_targets), value=0.0)
            .numpy() * 255.0
        ).astype(np.uint8)
        image_cond = {"img": all_imgs_np, "input_indices": list(range(num_inputs))}
        camera_cond = {"c2w": all_c2ws, "K": all_Ks}

        # Normalize intrinsics once
        for k in [0] + list(range(len(traj_prior_Ks))):
            K = camera_cond["K"][k].clone()
            K[0] /= W
            K[1] /= H
            camera_cond["K"][k] = K

        # --- Input images ---
        input_imgs = torch.as_tensor(image_cond["img"][0]).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
        traj_prior_imgs = torch.zeros(len(traj_prior_c2ws), *input_imgs.shape[1:], device=device)

        if not osp.exists(save_path):
            os.makedirs(save_path)

        save_output({"/image": input_imgs}, save_path=osp.join(save_path, "input"), video_save_fps=2)

        torch.cuda.empty_cache()
        seed_everything(seed)

        # --- Prepare batch ---
        curr_imgs = torch.cat([input_imgs, traj_prior_imgs], dim=0)
        curr_c2ws = torch.cat([input_c2ws, traj_prior_c2ws], dim=0)
        curr_Ks = torch.cat([input_Ks, traj_prior_Ks], dim=0)

        value_dict = get_value_dict(
            curr_imgs.to(device),
            [0],
            curr_c2ws,
            curr_Ks,
            list(range(T)),
            all_c2ws=camera_cond["c2w"],
            camera_scale=camera_scale,
        )

        # --- Sampler ---
        sampler = EulerEDMSampler(
            discretization=denoiser.discretization,
            guider=MultiviewCFG(cfg_min=1.2),
            num_steps=50,
            device=device,
        )

        imgs = value_dict["cond_frames"].to(device)
        input_masks = value_dict["cond_frames_mask"].to(device)
        pluckers = value_dict["plucker_coordinate"].to(device)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            latents = torch.nn.functional.pad(ae.encode(imgs[input_masks], 1), (0, 0, 0, 0, 0, 1), value=1.0)

            c_crossattn = repeat(conditioner(imgs[input_masks]).mean(0), "d -> n 1 d", n=T)
            uc_crossattn = torch.zeros_like(c_crossattn)

            c_replace = torch.zeros(T, *latents.shape[1:], device=latents.device)
            c_replace[input_masks] = latents

            c_concat = torch.cat([repeat(input_masks, "n -> n 1 h w", h=pluckers.shape[2], w=pluckers.shape[3]), pluckers], 1)
            uc_concat = torch.cat([torch.zeros(T, 1, *pluckers.shape[-2:], device=pluckers.device), pluckers], 1)

            c = {"crossattn": c_crossattn, "replace": c_replace, "concat": c_concat, "dense_vector": pluckers}
            uc = {"crossattn": uc_crossattn, "replace": torch.zeros_like(c_replace), "concat": uc_concat, "dense_vector": pluckers}

            randn = torch.randn((T, 4, H // 8, W // 8), device=device)
            samples_z = sampler(
                lambda x, sigma, c: denoiser(model, x, sigma, c, num_frames=T),
                randn,
                scale=float(cfg),
                cond=c,
                uc=uc,
                c2w=value_dict["c2w"].to(device),
                K=value_dict["K"].to(device),
                input_frame_mask=value_dict["cond_frames_mask"].to(device),
            )
            samples = ae.decode(samples_z, 1)
            samples = decode_output(samples[1:], T)

            save_output(samples, save_path=osp.join(save_path, "first-pass"), video_save_fps=5)


def main(
    input_img: str,
    seed: int = 23,
    cfg: float = 4.0,
    preset_traj: str = "orbit",
    num_frames: int = 20,
    zoom_factor: float = 1.0,
    camera_scale: float = 2.0,
):
    renderer = SevaRenderer()
    preprocessed = preprocess(input_img)
    renderer.render(preprocessed, seed, cfg, preset_traj, num_frames, zoom_factor, camera_scale)

    print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Peak memory reserved:  {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    free, total = torch.cuda.mem_get_info()
    print(f"Free: {free/1024**3:.2f} GB, Total: {total/1024**3:.2f} GB")


if __name__ == "__main__":
    tyro.cli(main)
