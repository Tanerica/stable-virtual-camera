import copy
import os.path as osp
from datetime import datetime

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro

from seva.eval import (
    infer_prior_stats,
    run_one_scene,
    transform_img_and_K,
)
from seva.geometry import DEFAULT_FOV_RAD, get_default_intrinsics, get_preset_pose_fov
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DiscreteDenoiser
from seva.utils import load_model

device = "cuda:0"

# Constants
WORK_DIR = "work_dirs/demo_cli"
MODEL = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
AE = AutoEncoder(chunk_size=1).to(device)
CONDITIONER = CLIPConditioner().to(device)
DENOISER = DiscreteDenoiser(num_idx=1000, device=device)

VERSION_DICT = {
    "H": 576,
    "W": 576,
    "T": 21,
    "C": 4,
    "f": 8,
    "options": {},
}


class SevaRenderer:
    def preprocess(self, input_img_path: str):
        shorter = 576
        shorter = round(shorter / 64) * 64

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
        self, preprocessed: dict, preset_traj: str, num_frames: int, zoom_factor: float
    ):
        img_wh = preprocessed["input_wh"]
        start_c2w = preprocessed["input_c2ws"][0]
        start_w2c = torch.linalg.inv(start_c2w)
        look_at = torch.tensor([0, 0, 10])
        start_fov = DEFAULT_FOV_RAD #DEFAULT_FOV_RAD = 0.9424777960769379  # 54 degrees by default

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

    def render(
        self,
        preprocessed: dict,
        seed: int,
        chunk_strategy: str,
        cfg: float,
        preset_traj: str,
        num_frames: int,
        zoom_factor: float,
        camera_scale: float,
    ):
        render_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        render_dir = osp.join(WORK_DIR, render_name)

        input_imgs, input_Ks, input_c2ws, (W, H) = (
            preprocessed["input_imgs"],
            preprocessed["input_Ks"],
            preprocessed["input_c2ws"],
            preprocessed["input_wh"],
        )

        num_inputs = len(input_imgs)
        input_c2ws = torch.eye(4)[None].to(dtype=input_c2ws.dtype)
        target_c2ws, target_Ks = self.get_target_c2ws_and_Ks_from_preset(
            preprocessed, preset_traj, num_frames, zoom_factor
        )

        all_c2ws = torch.cat([input_c2ws, target_c2ws], 0)
        all_Ks = torch.cat([input_Ks, target_Ks], 0) * input_Ks.new_tensor([W, H, 1])[
            :, None
        ]

        num_targets = len(target_c2ws)
        input_indices = list(range(num_inputs))
        target_indices = np.arange(num_inputs, num_inputs + num_targets).tolist()

        # Get anchor cameras
        T = VERSION_DICT["T"]
        version_dict = copy.deepcopy(VERSION_DICT)
        num_anchors = infer_prior_stats(
            T, num_inputs, num_total_frames=num_targets, version_dict=version_dict
        )
        T = version_dict["T"]

        anchor_indices = np.linspace(
            num_inputs, num_inputs + num_targets - 1, num_anchors
        ).tolist()
        anchor_c2ws = all_c2ws[[round(ind) for ind in anchor_indices]]
        anchor_Ks = all_Ks[[round(ind) for ind in anchor_indices]]

        # Conditioning
        all_imgs_np = (
            F.pad(input_imgs, (0, 0, 0, 0, 0, 0, 0, num_targets), value=0.0).numpy()
            * 255.0
        ).astype(np.uint8)
        image_cond = {
            "img": all_imgs_np,
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
        }
        camera_cond = {
            "c2w": all_c2ws,
            "K": all_Ks,
            "input_indices": list(range(num_inputs + num_targets)),
        }

        # Options
        num_steps = 50
        options = copy.deepcopy(VERSION_DICT["options"])
        options["chunk_strategy"] = chunk_strategy
        options["video_save_fps"] = 30.0
        options["beta_linear_start"] = 5e-6
        options["log_snr_shift"] = 2.4
        options["guider_types"] = [1, 2]
        options["cfg"] = [float(cfg), 3.0 if num_inputs >= 9 else 2.0]
        options["camera_scale"] = camera_scale
        options["num_steps"] = num_steps
        options["cfg_min"] = 1.2
        options["encoding_t"] = 1
        options["decoding_t"] = 1

        print("Rendering started...")
        video_path_generator = run_one_scene(
            task="img2trajvid",
            version_dict={
                "H": H,
                "W": W,
                "T": T,
                "C": VERSION_DICT["C"],
                "f": VERSION_DICT["f"],
                "options": options,
            },
            model=MODEL,
            ae=AE,
            conditioner=CONDITIONER,
            denoiser=DENOISER,
            image_cond=image_cond,
            camera_cond=camera_cond,
            save_path=render_dir,
            use_traj_prior=True,
            traj_prior_c2ws=anchor_c2ws,
            traj_prior_Ks=anchor_Ks,
            seed=seed,
            gradio=False,
        )

        for i, video_path in enumerate(video_path_generator):
            print(f"Generated video {i}: {video_path}")


def main(
    input_img: str,
    seed: int = 23,
    chunk_strategy: str = "interp",
    cfg: float = 4.0,
    preset_traj: str = "orbit",
    num_frames: int = 80,
    zoom_factor: float = 1.0,
    camera_scale: float = 2.0,
):
    renderer = SevaRenderer()
    preprocessed = renderer.preprocess(input_img)
    renderer.render(
        preprocessed,
        seed,
        chunk_strategy,
        cfg,
        preset_traj,
        num_frames,
        zoom_factor,
        camera_scale,
    )


if __name__ == "__main__":
    tyro.cli(main)
