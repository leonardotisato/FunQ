import torch
import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 classifier.

    Args:
        nr_classes (int): number of output classes.
        dropout (float): dropout probability applied before the final FC layer.
        pretrained (bool): if True, loads ImageNet pretrained weights.
    """

    def __init__(self, nr_classes: int = 2, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        in_features = backbone.fc.in_features

        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, nr_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        features = self.backbone(x)      # shape: [B, in_features]
        features = self.dropout(features)
        logits = self.fc(features)       # shape: [B, nr_classes]
        return logits
    

    def replace_head(self, nr_classes: int) -> None:
        """Replace only the final classifier (self.fc) keeping same in_features & dropout."""
        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, nr_classes)



class ResNet18Classifier(nn.Module):
    """
    ResNet-18 classifier.

    Args:
        nr_classes (int): number of output classes.
        dropout (float): dropout probability applied before the final FC layer.
        pretrained (bool): if True, loads ImageNet pretrained weights.
    """

    def __init__(self, nr_classes: int = 2, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = backbone.fc.in_features

        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, nr_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      
        features = self.backbone(x)      # shape: [B, in_features]
        features = self.dropout(features)
        logits = self.fc(features)       # shape: [B, nr_classes]
        return logits

# -------------------------
# VGG16
# -------------------------
class VGG16Classifier(nn.Module):
    def __init__(self, nr_classes: int = 4, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()

        backbone = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        )

        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, nr_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)      # [B, in_features]
        features = self.dropout(features)
        logits = self.fc(features)
        return logits

    def replace_head(self, nr_classes: int) -> None:
        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, nr_classes)


# -------------------------
# ViT (vit_b_16)
# -------------------------
class ViTClassifier(nn.Module):
    def __init__(self, nr_classes: int = 4, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()

        backbone = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        )

        in_features = backbone.heads.head.in_features
        backbone.heads.head = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, nr_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)      # [B, in_features]
        features = self.dropout(features)
        logits = self.fc(features)
        return logits

    def replace_head(self, nr_classes: int) -> None:
        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, nr_classes)


# -------------------------
# Swin Transformer (swin_t)
# -------------------------
class SwinClassifier(nn.Module):
    def __init__(self, nr_classes: int = 4, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()

        backbone = models.swin_t(
            weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        )

        in_features = backbone.head.in_features
        backbone.head = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, nr_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)      # [B, in_features]
        features = self.dropout(features)
        logits = self.fc(features)
        return logits

    def replace_head(self, nr_classes: int) -> None:
        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, nr_classes)


# -------------------------
# ConvNeXt Tiny
# -------------------------
class ConvNeXtTinyClassifier(nn.Module):
    def __init__(self, nr_classes: int = 4, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()

        backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        )

        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, nr_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)      # [B, in_features]
        features = self.dropout(features)
        logits = self.fc(features)
        return logits

    def replace_head(self, nr_classes: int) -> None:
        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, nr_classes)
