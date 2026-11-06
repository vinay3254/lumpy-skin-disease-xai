import timm
import torch
from torch import Tensor, nn


class HybridNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mode: str = "hybrid",
        fusion: str = "attention",
        pretrained: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.fusion = fusion

        if mode in ("hybrid", "cnn"):
            self.cnn = timm.create_model(
                "efficientnet_b0", pretrained=pretrained, num_classes=0
            )
            self.cnn_proj = nn.Linear(1280, 256)
        if mode in ("hybrid", "vit"):
            self.vit = timm.create_model(
                "deit_tiny_patch16_224", pretrained=pretrained, num_classes=0
            )
            self.vit_proj = nn.Linear(192, 256)
        if mode == "hybrid" and fusion == "attention":
            self.attention = nn.MultiheadAttention(256, 4, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(512 if mode == "hybrid" else 256, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    @property
    def gradcam_layer(self):
        return None if self.mode == "vit" else self.cnn.conv_head

    def forward(self, x: Tensor) -> Tensor:
        if self.mode in ("hybrid", "cnn"):
            cnn_tokens = self.cnn.forward_features(x).flatten(2).transpose(1, 2)
            cnn_tokens = self.cnn_proj(cnn_tokens)
            cnn_vector = cnn_tokens.mean(dim=1)
        if self.mode in ("hybrid", "vit"):
            vit_tokens = self.vit_proj(self.vit.forward_features(x))
            vit_vector = vit_tokens[:, 0]

        if self.mode == "cnn":
            features = cnn_vector
        elif self.mode == "vit":
            features = vit_vector
        else:
            if self.fusion == "attention":
                attended, _ = self.attention(
                    cnn_tokens, vit_tokens, vit_tokens, need_weights=False
                )
                cnn_vector = attended.mean(dim=1)
            features = torch.cat((cnn_vector, vit_vector), dim=1)
        return self.head(features)


def build_model(mode, fusion, num_classes, pretrained=True) -> HybridNet:
    return HybridNet(num_classes, mode=mode, fusion=fusion, pretrained=pretrained)
