from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from diffusers.models.modeling_utils import ModelMixin
from seva.modules.layers import (
    Downsample,
    GroupNorm32,
    ResBlock,
    TimestepEmbedSequential,
    Upsample,
    timestep_embedding,
)
from seva.modules.transformer import MultiviewTransformer


@dataclass
class SevaParams(object):
    in_channels: int = 11
    model_channels: int = 320
    out_channels: int = 4
    num_frames: int = 21
    num_res_blocks: int = 2
    attention_resolutions: list[int] = field(default_factory=lambda: [4, 2, 1])
    channel_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_head_channels: int = 64
    transformer_depth: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    context_dim: int = 1024
    dense_in_channels: int = 6
    dropout: float = 0.0
    unflatten_names: list[str] = field(
        default_factory=lambda: ["middle_ds8", "output_ds4", "output_ds2"]
    )

    def __post_init__(self):
        assert len(self.channel_mult) == len(self.transformer_depth)


class Seva_ori(nn.Module):
    def __init__(self, params: SevaParams) -> None:
        super().__init__()
        self.params = params
        self.model_channels = params.model_channels
        self.out_channels = params.out_channels
        self.num_head_channels = params.num_head_channels

        time_embed_dim = params.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(params.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(params.in_channels, params.model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = params.model_channels
        input_block_chans = [params.model_channels]
        ch = params.model_channels
        ds = 1
        for level, mult in enumerate(params.channel_mult):
            for _ in range(params.num_res_blocks):
                input_layers: list[ResBlock | MultiviewTransformer | Downsample] = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        out_channels=mult * params.model_channels,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = mult * params.model_channels
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels
                    input_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"input_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*input_layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(params.channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        num_heads = ch // params.num_head_channels
        dim_head = params.num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
            MultiviewTransformer(
                ch,
                num_heads,
                dim_head,
                name=f"middle_ds{ds}",
                depth=params.transformer_depth[-1],
                context_dim=params.context_dim,
                unflatten_names=params.unflatten_names,
            ),
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(params.channel_mult))[::-1]:
            for i in range(params.num_res_blocks + 1):
                ich = input_block_chans.pop()
                output_layers: list[ResBlock | MultiviewTransformer | Upsample] = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        out_channels=params.model_channels * mult,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = params.model_channels * mult
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels

                    output_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"output_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                        )
                    )
                if level and i == params.num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    output_layers.append(Upsample(ch, out_ch))
                self.output_blocks.append(TimestepEmbedSequential(*output_layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, params.out_channels, 3, padding=1),
        )
        

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
        )
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
            )
        h = h.type(x.dtype)
        return self.out(h)

class Seva2(ModelMixin):
    _supports_gradient_checkpointing = True
    def __init__(self, params: SevaParams) -> None:
        super().__init__()
        self.params = params
        self.model_channels = params.model_channels
        self.out_channels = params.out_channels
        self.num_head_channels = params.num_head_channels

        time_embed_dim = params.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(params.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(params.in_channels, params.model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = params.model_channels
        input_block_chans = [params.model_channels]
        ch = params.model_channels
        ds = 1
        for level, mult in enumerate(params.channel_mult):
            for _ in range(params.num_res_blocks):
                input_layers: list[ResBlock | MultiviewTransformer | Downsample] = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        out_channels=mult * params.model_channels,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = mult * params.model_channels
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels
                    input_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"input_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*input_layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(params.channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        num_heads = ch // params.num_head_channels
        dim_head = params.num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
            MultiviewTransformer(
                ch,
                num_heads,
                dim_head,
                name=f"middle_ds{ds}",
                depth=params.transformer_depth[-1],
                context_dim=params.context_dim,
                unflatten_names=params.unflatten_names,
            ),
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(params.channel_mult))[::-1]:
            for i in range(params.num_res_blocks + 1):
                ich = input_block_chans.pop()
                output_layers: list[ResBlock | MultiviewTransformer | Upsample] = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        out_channels=params.model_channels * mult,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = params.model_channels * mult
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels

                    output_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"output_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                        )
                    )
                if level and i == params.num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    output_layers.append(Upsample(ch, out_ch))
                self.output_blocks.append(TimestepEmbedSequential(*output_layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, params.out_channels, 3, padding=1),
        )
    # def _set_gradient_checkpointing(self, module, value=False):
    #     if hasattr(module, "gradient_checkpointing"):
    #         module.gradient_checkpointing = value
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        hs = []
        h = x

        # Apply gradient checkpointing to the input blocks
        for module in self.input_blocks:
            # Pass all arguments as a tuple to the checkpoint function
            h = checkpoint(module, h, t_emb, y, dense_y, num_frames)
            hs.append(h)

        # Apply gradient checkpointing to the middle block
        h = checkpoint(self.middle_block, h, t_emb, y, dense_y, num_frames)

        # Apply gradient checkpointing to the output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            # Pass all arguments as a tuple to the checkpoint function
            h = checkpoint(module, h, t_emb, y, dense_y, num_frames)

        return self.out(h)

class Seva4(ModelMixin):
    _supports_gradient_checkpointing = True
    def __init__(self, params: SevaParams) -> None:
        super().__init__()
        self.params = params
        self.model_channels = params.model_channels
        self.out_channels = params.out_channels
        self.num_head_channels = params.num_head_channels

        time_embed_dim = params.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(params.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(params.in_channels, params.model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = params.model_channels
        input_block_chans = [params.model_channels]
        ch = params.model_channels
        ds = 1
        for level, mult in enumerate(params.channel_mult):
            for _ in range(params.num_res_blocks):
                input_layers: list[ResBlock | MultiviewTransformer | Downsample] = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        out_channels=mult * params.model_channels,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = mult * params.model_channels
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels
                    input_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"input_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*input_layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(params.channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        num_heads = ch // params.num_head_channels
        dim_head = params.num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
            MultiviewTransformer(
                ch,
                num_heads,
                dim_head,
                name=f"middle_ds{ds}",
                depth=params.transformer_depth[-1],
                context_dim=params.context_dim,
                unflatten_names=params.unflatten_names,
            ),
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(params.channel_mult))[::-1]:
            for i in range(params.num_res_blocks + 1):
                ich = input_block_chans.pop()
                output_layers: list[ResBlock | MultiviewTransformer | Upsample] = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        out_channels=params.model_channels * mult,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = params.model_channels * mult
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels

                    output_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"output_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                        )
                    )
                if level and i == params.num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    output_layers.append(Upsample(ch, out_ch))
                self.output_blocks.append(TimestepEmbedSequential(*output_layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, params.out_channels, 3, padding=1),
        )
        self.gradient_checkpointing = True
    # def _set_gradient_checkpointing(self, module, value=False):
    #     if hasattr(module, "gradient_checkpointing"):
    #         module.gradient_checkpointing = value
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
        )
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
            )
        h = h.type(x.dtype)
        return self.out(h)


class Seva5(ModelMixin):
    _supports_gradient_checkpointing = True

    def __init__(self, params: SevaParams) -> None:
        super().__init__()
        self.params = params
        self.model_channels = params.model_channels
        self.out_channels = params.out_channels
        self.num_head_channels = params.num_head_channels
        self.gradient_checkpointing = False   # ← Required

        time_embed_dim = params.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(params.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # --------------------
        # Input blocks
        # --------------------
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(params.in_channels, params.model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = params.model_channels
        input_block_chans = [params.model_channels]
        ch = params.model_channels
        ds = 1

        for level, mult in enumerate(params.channel_mult):
            for _ in range(params.num_res_blocks):
                input_layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        out_channels=mult * params.model_channels,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = mult * params.model_channels

                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels
                    input_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"input_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*input_layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(params.channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        # --------------------
        # Middle block
        # --------------------
        num_heads = ch // params.num_head_channels
        dim_head = params.num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
            MultiviewTransformer(
                ch,
                num_heads,
                dim_head,
                name=f"middle_ds{ds}",
                depth=params.transformer_depth[-1],
                context_dim=params.context_dim,
                unflatten_names=params.unflatten_names,
            ),
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
        )
        self._feature_size += ch

        # --------------------
        # Output blocks
        # --------------------
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(params.channel_mult))[::-1]:
            for i in range(params.num_res_blocks + 1):
                ich = input_block_chans.pop()
                output_layers = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        out_channels=params.model_channels * mult,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = params.model_channels * mult

                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels
                    output_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"output_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                        )
                    )

                if level and i == params.num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    output_layers.append(Upsample(ch, out_ch))

                self.output_blocks.append(TimestepEmbedSequential(*output_layers))
                self._feature_size += ch

        # final output
        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, params.out_channels, 3, padding=1),
        )

    # ============================================================
    # HF Gradient Checkpointing API
    # ============================================================
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Seva):
            module.gradient_checkpointing = value

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        self.apply(lambda m: self._set_gradient_checkpointing(m, True))

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.apply(lambda m: self._set_gradient_checkpointing(m, False))

    # ============================================================
    # Forward
    # ============================================================
    def _maybe_checkpoint(self, module, *args):
        if self.gradient_checkpointing:
            return checkpoint(module, *args)
        return module(*args)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        
        num_frames = num_frames or self.params.num_frames

        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        hs = []
        h = x

        # Input blocks
        for module in self.input_blocks:
            h = self._maybe_checkpoint(module, h, t_emb, y, dense_y, num_frames)
            hs.append(h)

        # Middle block
        h = self._maybe_checkpoint(self.middle_block, h, t_emb, y, dense_y, num_frames)

        # Output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = self._maybe_checkpoint(module, h, t_emb, y, dense_y, num_frames)

        return self.out(h)


class Seva(ModelMixin):
    _supports_gradient_checkpointing = True

    def __init__(self, params: SevaParams) -> None:
        super().__init__()
        self.params = params
        self.model_channels = params.model_channels
        self.out_channels = params.out_channels
        self.num_head_channels = params.num_head_channels

        time_embed_dim = params.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(params.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(params.in_channels, params.model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = params.model_channels
        input_block_chans = [params.model_channels]
        ch = params.model_channels
        ds = 1
        for level, mult in enumerate(params.channel_mult):
            for _ in range(params.num_res_blocks):
                input_layers: list[ResBlock | MultiviewTransformer | Downsample] = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        out_channels=mult * params.model_channels,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = mult * params.model_channels
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels
                    input_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"input_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*input_layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(params.channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        num_heads = ch // params.num_head_channels
        dim_head = params.num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
            MultiviewTransformer(
                ch,
                num_heads,
                dim_head,
                name=f"middle_ds{ds}",
                depth=params.transformer_depth[-1],
                context_dim=params.context_dim,
                unflatten_names=params.unflatten_names,
            ),
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(params.channel_mult))[::-1]:
            for i in range(params.num_res_blocks + 1):
                ich = input_block_chans.pop()
                output_layers: list[ResBlock | MultiviewTransformer | Upsample] = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        out_channels=params.model_channels * mult,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = params.model_channels * mult
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels

                    output_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"output_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                        )
                    )
                if level and i == params.num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    output_layers.append(Upsample(ch, out_ch))
                self.output_blocks.append(TimestepEmbedSequential(*output_layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, params.out_channels, 3, padding=1),
        )

    # Diffusers-style model-wide setter (called by ModelMixin.enable_gradient_checkpointing)
    def _set_gradient_checkpointing(self, enable: bool = False, gradient_checkpointing_func=None):
        # store the checkpoint function provided by ModelMixin.enable_gradient_checkpointing
        # (ModelMixin passes this in as `gradient_checkpointing_func`)
        self._gradient_checkpointing_func = gradient_checkpointing_func

        # propagate the boolean flag to submodules we want to checkpoint
        for module in self.modules():
            if isinstance(module, (ResBlock, MultiviewTransformer, TimestepEmbedSequential)):
                module.gradient_checkpointing = enable

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        hs = []
        h = x
        # input blocks
        for module in self.input_blocks:
            if self.is_gradient_checkpointing and self.training:
                # use the injected checkpoint function from ModelMixin
                h = self._gradient_checkpointing_func(module, h, t_emb, y, dense_y, num_frames)
            else:
                h = module(
                    h,
                    emb=t_emb,
                    context=y,
                    dense_emb=dense_y,
                    num_frames=num_frames,
                )
            hs.append(h)

        # middle block
        if self.is_gradient_checkpointing and self.training:
            h = self._gradient_checkpointing_func(self.middle_block, h, t_emb, y, dense_y, num_frames)
        else:
            h = self.middle_block(
                h, emb=t_emb, context=y, dense_emb=dense_y, num_frames=num_frames
            )

        # output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if self.is_gradient_checkpointing and self.training:
                h = self._gradient_checkpointing_func(module, h, t_emb, y, dense_y, num_frames)
            else:
                h = module(
                    h,
                    emb=t_emb,
                    context=y,
                    dense_emb=dense_y,
                    num_frames=num_frames,
                )
        return self.out(h)


class SGMWrapper(nn.Module):
    def __init__(self, module: Seva):
        super().__init__()
        self.module = module

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.module(
            x,
            t=t,
            y=c["crossattn"],
            dense_y=c["dense_vector"],
            **kwargs,
        )
