from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from eb_jepa.nn_utils import TemporalBatchMixin, init_module_weights


class conv3d2(nn.Sequential):
    """Simple 3D convnet with 2 layers."""

    def __init__(self, in_d, h_d, out_d, tk, ts, sk, ss, pad):
        super(conv3d2, self).__init__(
            nn.Conv3d(
                in_d, h_d, kernel_size=(tk, sk, sk), stride=(1, 1, 1), padding=pad
            ),
            nn.ReLU(),
            nn.Conv3d(
                h_d, out_d, kernel_size=(tk, sk, sk), stride=(ts, ss, ss), padding=pad
            ),
        )
        self.apply(init_module_weights)
        self.input_dim = in_d
        self.hidden_dim = h_d
        self.output_dim = out_d
        # t_shift is the index (in the time dimension) of the first output
        # cannot see its coresponding input
        if pad == "valid":
            self.t_shift = 2 * tk - 1
        elif pad == "same":
            self.t_shift = 2 * (tk - 1)
        else:
            raise NameError("invalid padding for con3d2. Must be 'valid' or 'same'")


class ResidualBlock(nn.Module):
    """Standard residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet5(TemporalBatchMixin, nn.Module):
    """
    A lightweight ResNet with 5 layers (2 blocks).
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(self, in_d, h_d, out_d, s1=1, s2=1, s3=1, avg_pool=False):
        super().__init__()
        self.avg_pool = avg_pool
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(h_d, h_d, stride=s1)
        self.layer2 = ResidualBlock(h_d, h_d * 2, stride=s2)
        self.layer3 = ResidualBlock(h_d * 2, out_d, stride=s3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if avg_pool else torch.nn.Identity()

    def _forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        if self.avg_pool:
            out = out.flatten(1)
        return out


class SimplePredictor(nn.Module):
    """Wrapper that concatenates states and actions channel-wise before prediction."""

    def __init__(self, predictor, context_length):
        super().__init__()
        self.predictor = predictor
        self.is_rnn = predictor.is_rnn
        self.context_length = context_length

    def forward(self, x, a):
        return self.predictor(torch.cat([x, a], dim=1))


class StateOnlyPredictor(SimplePredictor):
    """Wrapper for a simple predictor which concatenates states and actions channel wise."""

    def forward(self, x, a):
        # action not used on purpose
        prev_state = x[:, :, :-1]  # [B, C, T-1, H, W]
        next_state = x[:, :, 1:]  # [B, C, T-1, H, W]
        combined_xa = torch.cat((prev_state, next_state), dim=1)
        return self.predictor(combined_xa)


class ResUNet(TemporalBatchMixin, nn.Module):
    """
    A small UNet with residual encoder blocks and transposed-conv upsampling.
    Channels scale like h, 2h, 4h, 8h. Output keeps the input HxW.
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(self, in_d, h_d, out_d, is_rnn=False):
        super().__init__()
        self.is_rnn = is_rnn
        # Stem
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)

        # Encoder
        self.enc1 = ResidualBlock(h_d, h_d, stride=1)  # H, W
        self.enc2 = ResidualBlock(h_d, 2 * h_d, stride=2)  # H/2, W/2
        self.enc3 = ResidualBlock(2 * h_d, 4 * h_d, stride=2)  # H/4, W/4
        self.bott = ResidualBlock(4 * h_d, 8 * h_d, stride=2)  # H/8, W/8

        # Decoder upsamples, then fuses skip with a residual block that reduces channels
        self.up3 = nn.ConvTranspose2d(8 * h_d, 4 * h_d, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(8 * h_d, 4 * h_d, stride=1)

        self.up2 = nn.ConvTranspose2d(4 * h_d, 2 * h_d, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(4 * h_d, 2 * h_d, stride=1)

        self.up1 = nn.ConvTranspose2d(2 * h_d, 1 * h_d, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(2 * h_d, 1 * h_d, stride=1)

        # Head
        self.head = nn.Conv2d(h_d, out_d, kernel_size=1)

    @staticmethod
    def _match_size(x, ref):
        # Guards against odd input sizes by resizing the upsample to the skip spatial dims
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(
                x, size=ref.shape[-2:], mode="bilinear", align_corners=False
            )
        return x

    def _forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))

        # Encoder with skips
        s1 = self.enc1(x0)  # h
        s2 = self.enc2(s1)  # 2h
        s3 = self.enc3(s2)  # 4h
        b = self.bott(s3)  # 8h

        # Decoder stage 3
        d3 = self.up3(b)
        d3 = self._match_size(d3, s3)
        d3 = torch.cat([d3, s3], dim=1)  # 4h + 4h = 8h
        d3 = self.dec3(d3)  # → 4h

        # Decoder stage 2
        d2 = self.up2(d3)
        d2 = self._match_size(d2, s2)
        d2 = torch.cat([d2, s2], dim=1)  # 2h + 2h = 4h
        d2 = self.dec2(d2)  # → 2h

        # Decoder stage 1
        d1 = self.up1(d2)
        d1 = self._match_size(d1, s1)
        d1 = torch.cat([d1, s1], dim=1)  # h + h = 2h
        d1 = self.dec1(d1)  # → h

        out = self.head(d1)  # → out_d channels
        return out


class Projector(nn.Module):
    """MLP projector built from a spec string like '256-512-128'."""

    def __init__(self, mlp_spec):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]  # Store output dimension as attribute

    def forward(self, x):
        return self.net(x)


class DetHead(nn.Module):
    """Detection head that pools features and predicts binary maps."""

    def __init__(self, in_d, h_d, out_d):
        super().__init__()
        self.head = nn.Sequential(conv3d2(in_d, h_d, out_d, 1, 1, 3, 1, "same"))
        self.apply(init_module_weights)

    def forward(self, x):
        """Forward pass on predictor output of shape (B, C, T, H, W)."""
        # (Batch, Feature, Time, Height, Width)
        # [8, 8, T, 8, 8]
        x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
        x = torch.stack(x, 2)
        # [8, T, 8, 8]
        x = self.head(x).squeeze(1)

        return torch.sigmoid(x)

    @torch.no_grad()
    def score(self, preds, targets):

        scores = []
        for T in range(len(preds) - 1):
            x = preds[T]
            x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
            x = torch.stack(x, 2)
            x = self.head(x).squeeze(1)

            y = targets[:, T:]
            x = x[:, T:]

            ap = average_precision_score(
                y.flatten().detach().long().cpu().numpy(),
                x.flatten().detach().cpu().numpy(),
                average="weighted",
            )
            scores.append(ap)

        return scores


class ResnetBlock(nn.Module):
    """ResNet Block."""

    def __init__(self, num_features):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)


class ResnetStack(nn.Module):
    """ResNet stack module."""

    def __init__(self, input_channels, num_features, num_blocks, max_pooling=True):
        super(ResnetStack, self).__init__()
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling
        self.initial_conv = nn.Conv2d(
            input_channels, num_features, kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList(
            [ResnetBlock(num_features) for _ in range(num_blocks)]
        )
        if max_pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.max_pool = nn.Identity()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
        return x


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    def __init__(
        self,
        width=1,
        stack_sizes=(16, 32, 32),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=2,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(2, 65, 65),
    ):
        super(ImpalaEncoder, self).__init__()
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.input_shape = input_shape
        self.mlp_output_dim = mlp_output_dim

        input_channels = [input_channels] + list(stack_sizes)

        self.stack_blocks = nn.ModuleList(
            [
                ResnetStack(
                    input_channels=input_channels[i],
                    num_features=stack_size * width,
                    num_blocks=num_blocks,
                )
                for i, stack_size in enumerate(stack_sizes)
            ]
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

        # Compute MLP input dimension dynamically
        with torch.no_grad():
            # Create a dummy input (assuming typical input size for this encoder)
            dummy_input = torch.zeros(1, *self.input_shape)  # (1, C, H, W)
            conv_out = dummy_input
            for stack_block in self.stack_blocks:
                conv_out = stack_block(conv_out)  # b c w h
            flattened_dim = conv_out.view(conv_out.size(0), -1).shape[1]  # c * w * h

        self.mlp = nn.Linear(flattened_dim, self.mlp_output_dim)

        if final_ln:
            self.final_ln = nn.LayerNorm(self.mlp_output_dim)
        else:
            self.final_ln = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            out: [B, D, T, 1, 1]
        """

        # [B, C, T, H, W] --> [T, B, C, H, W]
        (
            _,
            _,
            t,
            _,
            _,
        ) = x.shape
        x = x.permute(2, 0, 1, 3, 4)

        features = []

        for i in range(t):

            conv_out = x[i]

            for i, stack_block in enumerate(self.stack_blocks):
                conv_out = stack_block(conv_out)
                if self.dropout_rate is not None:
                    conv_out = self.dropout(conv_out)

            conv_out = F.relu(conv_out)
            if self.layer_norm:
                conv_out = nn.LayerNorm(conv_out.size()[1:])(conv_out)  # b c w h
            # flatten
            out = conv_out.view(conv_out.size(0), -1)
            out = self.mlp(out)
            out = self.final_ln(out)

            features.append(out)

        features = torch.stack(features, dim=1)

        features = features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)

        return features


class DINOv3ConvNextEncoder(nn.Module):
    """Pretrained DINOv3 ConvNeXt encoder with a learned spatial bottleneck.

    The official backbone returns one global token followed by flattened
    spatial tokens. We discard the global token, preserve all spatial tokens
    through a learned projection, and return the repository's standard
    ``[B,D,T,1,1]`` latent format.
    """

    def __init__(
        self,
        model_name: str,
        input_shape: tuple[int, int, int],
        latent_dim: int = 768,
        image_mean: tuple[float, float, float] = (
            0.485,
            0.456,
            0.406,
        ),
        image_std: tuple[float, float, float] = (
            0.229,
            0.224,
            0.225,
        ),
        frame_batch_size: int | None = 128,
        local_files_only: bool = False,
        revision: str | None = None,
        gradient_checkpointing: bool = False,
        load_pretrained: bool = True,
    ):
        super().__init__()
        try:
            from transformers import (
                AutoModel,
                DINOv3ConvNextConfig,
                DINOv3ConvNextModel,
            )
        except ImportError as error:  # pragma: no cover - dependency error.
            raise ImportError(
                "DINOv3ConvNextEncoder requires `transformers`. "
                "Install the project dependencies with `uv sync`."
            ) from error

        if input_shape[0] != 3:
            raise ValueError(
                "DINOv3 ConvNeXt expects RGB input, got " f"input_shape={input_shape}"
            )
        if frame_batch_size is not None and frame_batch_size <= 0:
            raise ValueError("frame_batch_size must be positive or null")

        load_kwargs = {"local_files_only": local_files_only}
        if revision is not None:
            load_kwargs["revision"] = revision
        if load_pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, **load_kwargs)
        else:
            if model_name != "facebook/dinov3-convnext-tiny-pretrain-lvd1689m":
                raise ValueError(
                    "Offline architecture construction is only defined for "
                    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
                )
            self.backbone = DINOv3ConvNextModel(
                DINOv3ConvNextConfig(image_size=input_shape[1])
            )
        if gradient_checkpointing:
            if not self.backbone.supports_gradient_checkpointing:
                raise ValueError(
                    "The Transformers DINOv3 ConvNeXt implementation does not "
                    "support gradient checkpointing"
                )
            self.backbone.gradient_checkpointing_enable()

        self.model_name = model_name
        self.input_shape = tuple(input_shape)
        self.mlp_output_dim = int(latent_dim)
        self.frame_batch_size = frame_batch_size
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )

        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            patch_tokens = self._extract_patch_tokens(self._normalize(dummy))
        self.backbone.train(was_training)

        self.patch_token_shape = tuple(patch_tokens.shape[1:])
        flattened_dim = patch_tokens[0].numel()
        self.spatial_projection = nn.Linear(flattened_dim, self.mlp_output_dim)
        self.final_ln = nn.LayerNorm(self.mlp_output_dim)
        init_module_weights(self.spatial_projection)
        init_module_weights(self.final_ln)

        self._trainable_backbone_stages = 0
        self.set_trainable_backbone_stages(0)

    def _normalize(self, frames: torch.Tensor) -> torch.Tensor:
        mean = self.image_mean.to(device=frames.device, dtype=frames.dtype)
        std = self.image_std.to(device=frames.device, dtype=frames.dtype)
        return (frames - mean) / std

    def _extract_patch_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=frames, return_dict=True)
        tokens = output.last_hidden_state
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise ValueError(
                "Expected DINOv3 ConvNeXt tokens [B,1+P,C], got "
                f"{tuple(tokens.shape)}"
            )
        return tokens[:, 1:]

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        yield from self.spatial_projection.parameters()
        yield from self.final_ln.parameters()

    @property
    def num_backbone_stages(self) -> int:
        return len(self.backbone.model.stages)

    @property
    def trainable_backbone_stages(self) -> int:
        return self._trainable_backbone_stages

    def set_trainable_backbone_stages(self, num_stages: int) -> bool:
        """Freeze the backbone except for its final ``num_stages`` stages."""
        if not 0 <= num_stages <= self.num_backbone_stages:
            raise ValueError(
                f"num_stages must be in [0,{self.num_backbone_stages}], "
                f"got {num_stages}"
            )
        changed = num_stages != self._trainable_backbone_stages
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if num_stages:
            for stage in self.backbone.model.stages[-num_stages:]:
                for parameter in stage.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.backbone.layer_norm.parameters():
                parameter.requires_grad_(True)
        self._trainable_backbone_stages = num_stages
        self.train(self.training)
        return changed

    def train(self, mode: bool = True):
        """Keep frozen ConvNeXt stages deterministic during JEPA training."""
        super().train(mode)
        if mode:
            self.backbone.eval()
            if self._trainable_backbone_stages:
                for stage in self.backbone.model.stages[
                    -self._trainable_backbone_stages :
                ]:
                    stage.train(True)
                self.backbone.layer_norm.train(True)
            self.spatial_projection.train(True)
            self.final_ln.train(True)
        return self

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 5:
            raise ValueError(
                "Expected video [B,C,T,H,W], got " f"{tuple(observations.shape)}"
            )
        batch_size, channels, timesteps, height, width = observations.shape
        expected = self.input_shape
        if (channels, height, width) != expected:
            raise ValueError(
                f"Expected frame shape {expected}, got " f"{(channels, height, width)}"
            )

        frames = (
            observations.permute(0, 2, 1, 3, 4)
            .reshape(batch_size * timesteps, channels, height, width)
            .contiguous()
        )
        chunk_size = self.frame_batch_size or frames.shape[0]
        features = []
        for chunk in frames.split(chunk_size):
            patch_tokens = self._extract_patch_tokens(self._normalize(chunk))
            if tuple(patch_tokens.shape[1:]) != self.patch_token_shape:
                raise ValueError(
                    "DINOv3 patch-token geometry changed from "
                    f"{self.patch_token_shape} to "
                    f"{tuple(patch_tokens.shape[1:])}"
                )
            projected = self.spatial_projection(patch_tokens.flatten(1))
            features.append(self.final_ln(projected))
        features = torch.cat(features, dim=0)
        return (
            features.view(batch_size, timesteps, self.mlp_output_dim)
            .transpose(1, 2)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .contiguous()
        )


class RNNPredictor(nn.Module):
    """GRU-based predictor for single-step state propagation."""

    def __init__(
        self,
        hidden_size: int = 512,
        action_dim: Optional[int] = 2,
        num_layers: int = 1,
        final_ln: Optional[torch.nn.Module] = None,
    ):
        super(RNNPredictor, self).__init__()

        self.num_layers = num_layers

        self.rnn = torch.nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.final_ln = final_ln
        self.is_rnn = True
        self.context_length = 0

    def forward(self, state, action):
        """
        Propagate one step forward.

        Args:
            state: [B, D, 1, 1, 1]
            action: [B, A, 1]
        Returns:
            next_state: [B, D, 1, 1, 1]
        """
        # This only does one step
        rnn_state = state.flatten(1, 4).unsqueeze(0).contiguous()  # [1, B, D]
        rnn_input = action.squeeze(-1).unsqueeze(0).contiguous()  # [1, B, A]

        next_state, _ = self.rnn(rnn_input, rnn_state)

        next_state = self.final_ln(next_state)

        return next_state[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def _adaln_modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply adaptive LayerNorm scale and shift parameters."""
    return x * (1 + scale) + shift


class CausalSelfAttention(nn.Module):
    """Pre-normalized causal self-attention used by the AR predictor."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dropout = float(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            tensor.view(tensor.shape[0], tensor.shape[1], self.heads, -1)
            .transpose(1, 2)
            .contiguous()
            for tensor in (q, k, v)
        )
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().flatten(2)
        return self.to_out(output)


class TransformerFeedForward(nn.Module):
    """Transformer MLP without an internal norm.

    Normalization is performed by the surrounding AdaLN block so the action
    modulation is not immediately normalized away.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionConditionedTransformerBlock(nn.Module):
    """Causal transformer block with per-token AdaLN-Zero conditioning."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = CausalSelfAttention(
            dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.mlp = TransformerFeedForward(dim, mlp_dim, dropout=dropout)
        self.attention_norm = nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.mlp_norm = nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

        # AdaLN-Zero starts each residual branch as the identity and learns how
        # strongly action conditioning should affect it.
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.adaln_modulation(condition).chunk(6, dim=-1)
        attention_input = _adaln_modulate(
            self.attention_norm(x),
            attention_shift,
            attention_scale,
        )
        x = x + attention_gate * self.attention(attention_input)
        mlp_input = _adaln_modulate(
            self.mlp_norm(x),
            mlp_shift,
            mlp_scale,
        )
        return x + mlp_gate * self.mlp(mlp_input)


class ActionSequenceEncoder(nn.Module):
    """Embed repo-format action sequences from ``[B,A,T]`` to ``[B,E,T]``."""

    def __init__(
        self,
        action_dim: int,
        embedding_dim: int,
        smoothed_dim: int | None = None,
        mlp_scale: float = 2.0,
    ):
        super().__init__()
        smoothed_dim = smoothed_dim or action_dim
        hidden_dim = max(embedding_dim, int(mlp_scale * embedding_dim))
        self.embedding_dim = int(embedding_dim)
        self.patch_embedding = nn.Conv1d(
            action_dim,
            smoothed_dim,
            kernel_size=1,
        )
        self.embedding = nn.Sequential(
            nn.Linear(smoothed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(
                "Actions must have shape [B,A,T], got " f"{tuple(actions.shape)}"
            )
        embedded = self.patch_embedding(actions.float()).transpose(1, 2)
        return self.embedding(embedded).transpose(1, 2).contiguous()


class ActionConditionedTransformerPredictor(nn.Module):
    """LeWorldModel-style causal latent predictor with AdaLN-Zero actions.

    The public interface follows the other predictors in this repository:
    states are ``[B,D,T,1,1]`` and encoded actions are ``[B,C,T]``. The
    predictor emits one next-latent prediction at every input timestep.
    """

    def __init__(
        self,
        state_dim: int,
        condition_dim: int,
        *,
        hidden_dim: int,
        depth: int,
        heads: int,
        dim_head: int = 64,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_seq_len: int = 17,
        history_size: int = 4,
    ):
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if history_size > max_seq_len:
            raise ValueError("history_size cannot exceed max_seq_len")
        if heads <= 0 or dim_head <= 0:
            raise ValueError("heads and dim_head must be positive")

        self.state_dim = int(state_dim)
        self.condition_dim = int(condition_dim)
        self.max_seq_len = int(max_seq_len)
        self.history_size = int(history_size)
        self.context_length = self.history_size
        self.initial_context_length = 1
        self.supports_sequence_prediction = True
        self.is_rnn = False

        self.state_projection = (
            nn.Linear(state_dim, hidden_dim)
            if state_dim != hidden_dim
            else nn.Identity()
        )
        self.condition_projection = (
            nn.Linear(condition_dim, hidden_dim)
            if condition_dim != hidden_dim
            else nn.Identity()
        )
        self.position_embedding = nn.Parameter(torch.empty(1, max_seq_len, hidden_dim))
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.blocks = nn.ModuleList(
            ActionConditionedTransformerBlock(
                hidden_dim,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim or 4 * hidden_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = (
            nn.Linear(hidden_dim, state_dim)
            if hidden_dim != state_dim
            else nn.Identity()
        )
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if states.ndim != 5:
            raise ValueError(
                "States must have shape [B,D,T,H,W], got " f"{tuple(states.shape)}"
            )
        if states.shape[-2:] != (1, 1):
            raise ValueError(
                "Transformer predictor requires 1x1 latent maps, got "
                f"{tuple(states.shape[-2:])}"
            )
        if actions is None or actions.ndim != 3:
            raise ValueError("Encoded actions must have shape [B,C,T]")
        if states.shape[0] != actions.shape[0] or states.shape[2] != actions.shape[2]:
            raise ValueError(
                "State and action sequences must share batch/time dimensions, got "
                f"{tuple(states.shape)} and {tuple(actions.shape)}"
            )
        timesteps = states.shape[2]
        if timesteps > self.max_seq_len:
            raise ValueError(
                f"Sequence length {timesteps} exceeds max_seq_len={self.max_seq_len}"
            )

        x = states[:, :, :, 0, 0].transpose(1, 2)
        condition = actions.transpose(1, 2)
        x = self.state_projection(x)
        condition = self.condition_projection(condition)
        x = self.embedding_dropout(
            x + self.position_embedding[:, :timesteps].to(dtype=x.dtype)
        )
        for block in self.blocks:
            x = block(x, condition)
        x = self.output_projection(self.final_norm(x))
        return x.transpose(1, 2).unsqueeze(-1).unsqueeze(-1).contiguous()


class InverseDynamicsModel(nn.Module):
    """
    Predicts the action that caused a transition from state_t to state_t_plus_1.
    Used as auxiliary task for representation learning.
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(init_module_weights)

    def forward(self, state_t, state_t_plus_1):
        """
        Args:
            state_t: State at time t, shape [B, D]
            state_t_plus_1: State at time t+1, shape [B, D]
        Returns:
            predicted_action: Action predicted to transform state_t to state_t_plus_1, shape [B, A]
        """
        combined_states = torch.cat([state_t, state_t_plus_1], dim=1)
        return self.model(combined_states)
