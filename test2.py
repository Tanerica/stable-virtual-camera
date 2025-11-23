from diffusers import UNet2DConditionModel
a = UNet2DConditionModel()
a.enable_gradient_checkpointing()