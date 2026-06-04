"""
Symmetric InfoNCE (NT-Xent) contrastive loss, the same objective as CLIP.

For a batch of N (image, text) pairs:
- Images and texts belonging to the same pair are positives (diagonal).
- All other combinations in the batch are negatives.
- Loss is symmetric: image->text + text->image, averaged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCELoss(nn.Module):
    """Symmetric cross-entropy over cosine similarities, scaled by temperature."""

    def __init__(self):
        super().__init__()

    def forward(self,
                image_features: torch.Tensor,   # (N, D), L2-normalized
                text_features:  torch.Tensor,   # (N, D), L2-normalized
                logit_scale:    torch.Tensor    # scalar (>0), exp of model's logit_scale
                ) -> torch.Tensor:
        # logits[i, j] = scale * <img_i, txt_j>
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text  = logit_scale * text_features  @ image_features.T

        labels = torch.arange(len(image_features), device=image_features.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text,  labels)
        return 0.5 * (loss_i2t + loss_t2i)
