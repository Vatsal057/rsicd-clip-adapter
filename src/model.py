"""
Adapter module and wrapped CLIP model.

Architecture:
    CLIPAdapterModel wraps a frozen CLIP model.
    Two lightweight MLP adapters sit on top of the image and text encoders.
    Only the adapters are trained. CLIP weights are completely frozen.

Adapter design (Houlsby-style bottleneck MLP):
    Input (D) -> LayerNorm -> Linear(D -> hidden_dim) -> GELU -> Dropout
              -> Linear(hidden_dim -> D) -> + residual
    D = 512 for ViT-B/32
"""

import torch
import torch.nn as nn
import open_clip


class BottleneckAdapter(nn.Module):
    """
    Lightweight bottleneck-MLP adapter placed AFTER a frozen CLIP encoder.

    Residual + near-zero up_proj init means the adapter starts as an
    approximate identity, so the first few optimizer steps don't
    catastrophically distort CLIP's pretrained features.
    """

    def __init__(self,
                 input_dim:  int   = 512,
                 hidden_dim: int   = 256,
                 dropout:    float = 0.1,
                 use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual

        self.layer_norm = nn.LayerNorm(input_dim)
        self.down_proj  = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout    = nn.Dropout(dropout)
        self.up_proj    = nn.Linear(hidden_dim, input_dim)

        # Near-identity init: up_proj is zero, down_proj is small.
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x + residual if self.use_residual else x


class CLIPAdapterModel(nn.Module):
    """
    Frozen CLIP + trainable dual bottleneck adapters.

    image_embedding = adapter_image(clip.encode_image(image))
    text_embedding  = adapter_text (clip.encode_text(text))

    Both outputs are L2-normalized inside `encode_image` / `encode_text`,
    so they're directly usable by FAISS or a contrastive loss without
    any further normalization.
    """

    def __init__(self,
                 clip_model_name: str  = "ViT-B-32",
                 clip_pretrained:  str  = "openai",
                 hidden_dim:       int  = 256,
                 dropout:          float = 0.1,
                 adapter_on_image: bool = True,
                 adapter_on_text:  bool = True,
                 use_residual:     bool = True,
                 learn_logit_scale: bool = True):
        super().__init__()

        # Load CLIP backbone
        self.clip, _ = open_clip.create_model_from_pretrained(
            clip_model_name, pretrained=clip_pretrained, force_quick_gelu=True
        )

        # Identify embedding dim. For ViT-B/32 this is 512.
        try:
            clip_dim = self.clip.visual.output_dim
        except AttributeError:
            clip_dim = self.clip.text_projection.shape[1]

        # Freeze ALL CLIP parameters
        for param in self.clip.parameters():
            param.requires_grad = False

        # Trainable adapters
        if adapter_on_image:
            self.adapter_image = BottleneckAdapter(clip_dim, hidden_dim, dropout, use_residual)
        else:
            self.adapter_image = nn.Identity()

        if adapter_on_text:
            self.adapter_text = BottleneckAdapter(clip_dim, hidden_dim, dropout, use_residual)
        else:
            self.adapter_text = nn.Identity()

        # Learnable logit_scale (temperature inverse). Initialized to
        # log(1/0.07) = 2.6592, matching OpenAI's CLIP.
        if learn_logit_scale:
            self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        else:
            self.register_buffer("logit_scale", torch.tensor(2.6592))

        self.clip_dim      = clip_dim
        self.adapter_on_image = adapter_on_image
        self.adapter_on_text  = adapter_on_text

    @torch.no_grad()
    def _clip_encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_image(image)

    @torch.no_grad()
    def _clip_encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_text(text)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images through frozen CLIP, then through the image adapter."""
        feats = self._clip_encode_image(image)
        feats = feats.float()
        feats = self.adapter_image(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode captions through frozen CLIP, then through the text adapter."""
        feats = self._clip_encode_text(text)
        feats = feats.float()
        feats = self.adapter_text(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats

    def forward(self,
                images: torch.Tensor,
                texts:  torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass.
        Returns (image_features, text_features, logit_scale).
        Both feature tensors are L2-normalized.
        """
        img_feats = self.encode_image(images)
        txt_feats = self.encode_text(texts)
        return img_feats, txt_feats, self.logit_scale.exp()

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_adapter_model(checkpoint_path: str,
                      device: str = "cpu",
                      strict: bool = True) -> CLIPAdapterModel:
    """Load a trained adapter model from a checkpoint saved by `train.py`."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    # Always set use_residual=True on construction; load_state_dict with
    # strict=False handles the no-residual case.
    cfg.setdefault("use_residual", True)
    model = CLIPAdapterModel(**cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    model.to(device)
    model.eval()
    return model
