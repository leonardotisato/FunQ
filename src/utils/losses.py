import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


# ==============================================
# KD Loss based on Kullback-Leibler Divergence
# ==============================================

def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float = 1.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """
    Standard knowledge distillation loss from
    'Distilling the Knowledge in a Neural Network' (Hinton et al.)

    student_logits: [B, C]
    teacher_logits: [B, C]
    T: temperature
    """
    # Student log-probabilities (softened)
    log_p_student = F.log_softmax(student_logits / T, dim=1)
    # Teacher probabilities (softened)
    p_teacher = F.softmax(teacher_logits / T, dim=1)

    # KL divergence tra distribuzioni del teacher e dello student
    loss_kd = F.kl_div(log_p_student, p_teacher, reduction=reduction) * (T * T)
    return loss_kd



'''# =======================================
# 2) MULTI-SCALE FEATURE KD (intermediate)
# =======================================

def multiscale_feature_kd_loss(
    feats_S: Dict[str, torch.Tensor],
    feats_T: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Loss MSE sui feature map intermedi (multi-scale KD).

    feats_S, feats_T: dict con chiavi tipo 'early', 'mid', 'deep'
                      e tensori [B, C, H, W]
    weights: pesi per ciascun livello, es:
             {'early':0.3, 'mid':0.5, 'deep':0.2}
    """
    if weights is None:
        weights = {k: 1.0 for k in feats_S.keys()}

    loss_ms = 0.0
    for key in feats_S.keys():
        fS = feats_S[key]
        fT = feats_T[key]

        # match spatial size
        if fS.shape[-2:] != fT.shape[-2:]:
            fT = F.interpolate(
                fT, size=fS.shape[-2:], mode="bilinear", align_corners=False
            )
        loss_ms = loss_ms + weights.get(key, 1.0) * F.mse_loss(fS, fT)

    return loss_ms


import torch
import torch.nn.functional as F


def masked_kd_logits_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    cam_T: torch.Tensor,
    T: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Masked / saliency-weighted KD sui logit.

    Args:
        student_logits: [B, C]
        teacher_logits: [B, C]
        cam_T: [B, H, W] CAM (o saliency map) del teacher, normalizzata in [0,1]
        T: temperatura per la KD
        eps: piccolo termine per stabilità numerica

    Ritorna:
        loss scalare (media sui sample, pesata da CAM)
    """

    # --- 1. KD standard (per-sample) ---
    log_probs_S = F.log_softmax(student_logits / T, dim=1)   # [B, C]
    probs_T     = F.softmax(teacher_logits / T, dim=1)       # [B, C]

    # KL per sample (senza media sul batch)
    # reduction='none' -> [B, C]
    kl_per_class  = F.kl_div(log_probs_S, probs_T, reduction="none")
    kl_per_sample = kl_per_class.sum(dim=1)                  # [B]

    # --- 2. Peso per immagine derivato dalla CAM del teacher ---
    # cam_T: [B, H, W] -> media spaziale [B]
    mask_flat   = cam_T.view(cam_T.size(0), -1)              # [B, H*W]
    mask_weight = mask_flat.mean(dim=1)                      # [B]

    # Clamp giusto per sicurezza
    mask_weight = torch.clamp(mask_weight, min=0.0, max=1.0)

    # --- 3. Loss pesata e media sul batch ---
    # IMPORTANTE: pesi normalizzati per non cambiare la scala media troppo
    norm_factor = mask_weight.mean() + eps
    weighted_kl = (mask_weight * kl_per_sample) / norm_factor

    loss = (weighted_kl.mean()) * (T * T)
    return loss



# ==============================================
# 5) LOSS COMPLETA: CE + KD + MS + MASK + CAM
# ==============================================

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def full_kd_loss_from_outputs(
    logits_S: torch.Tensor,
    logits_T: torch.Tensor,
    feats_S: Dict[str, torch.Tensor],
    feats_T: Dict[str, torch.Tensor],
    y: torch.Tensor,
    compute_cam,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    delta: float = 0.5,
    temperature: float = 1.0,
    ms_weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Implementa la combinazione:

    loss_total = alpha * CE +
                 beta  * KD_logits +
                 gamma * KD_multiscale +
                 delta * (KD_masked + CAM_consistency)

    Parametri:
      - logits_S, logits_T: [B, C]
      - feats_S, feats_T: dict di feature map intermedi
      - y: label [B]
      - compute_cam: funzione (logits, feat_mid) -> CAM [B, H, W]
      - alpha, beta, gamma, delta, temperature: iperparametri della loss
      - ms_weights: pesi dei livelli per multi-scale KD

    Ritorna:
      loss_total, dict con tutte le componenti.
    """

    # ------- 1. Cross-Entropy -------
    loss_ce = F.cross_entropy(logits_S, y)

    # ------- 2. KD standard sui logit -------
    loss_kd = kd_standard_loss(
        student_logits=logits_S,
        teacher_logits=logits_T,
        T=temperature,
        reduction="batchmean",
    )

    # ------- 3. Multi-scale KD (intermediate features) -------
    loss_ms = multiscale_feature_kd_loss(
        feats_S=feats_S,
        feats_T=feats_T,
        weights=ms_weights
        if ms_weights is not None
        else {"early": 0.3, "mid": 0.5, "deep": 0.2},
    )

    # ------- 4. CAM del teacher / student -------
    cam_T = compute_cam(logits_T, feats_T["mid"])  # [B, H, W]
    cam_S = compute_cam(logits_S, feats_S["mid"])  # [B, H, W]

    # ------- 5. Masked KD (saliency-weighted KD sui logit) -------
    loss_masked_kd = masked_kd_logits_loss(
        student_logits=logits_S,
        teacher_logits=logits_T,
        cam_T=cam_T,
        T=temperature,
    )

    # ------- 6. CAM consistency (MSE tra CAM S/T) -------
    loss_cam = cam_consistency_loss(
        cam_S=cam_S,
        cam_T=cam_T,
        mode="mse",
    )

    # ------- 7. TOTAL LOSS -------
    loss_total = (
        alpha * loss_ce
        + beta * loss_kd
        + gamma * loss_ms
        + delta * (loss_masked_kd + loss_cam)
    )

    components = {
        "ce": loss_ce.detach(),
        "kd_logits": loss_kd.detach(),
        "kd_multiscale": loss_ms.detach(),
        "kd_masked": loss_masked_kd.detach(),
        "cam_consistency": loss_cam.detach(),
    }

    return loss_total, components




def compute_cam_weight(cam, logits, alpha = 1.0, beta = 1.0):
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)

    mean_cam = cam.mean()

    conf = torch.softmax(logits, dim = 0).max()

    weight = (mean_cam ** alpha) * (conf ** beta)
    return weight.detach()


loss_contrastive = ((z_S - z_T)**2).sum(dim= 1)
loss_contrastive_weighted = (cam_weight.squeeze() * loss_contrastive).mean()'''