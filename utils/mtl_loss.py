import torch
import torch.nn as nn
from utils.losses import CrossEntropy2d


class HydraNetMTLoss(nn.Module):
    """
    Multi-task uncertainty weighting loss for segmentation (multi-class) and edges (binary).

    L = 0.5*exp(-s1)*L_seg + 0.5*s1 + 0.5*exp(-s2)*L_edge + 0.5*s2
    where s_i = log(sigma_i^2) are learnable.
    """

    def __init__(self, seg_criterion: nn.Module | None = None, edge_criterion: nn.Module | None = None):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))
        self.seg_criterion = seg_criterion if seg_criterion is not None else CrossEntropy2d()
        self.edge_criterion = edge_criterion if edge_criterion is not None else nn.BCEWithLogitsLoss()

    def forward(self, seg_logits: torch.Tensor, edge_logits: torch.Tensor | None,
                seg_targets: torch.Tensor, edge_targets: torch.Tensor | None = None):
        device = seg_logits.device
        loss_seg = self.seg_criterion(seg_logits, seg_targets)

        loss_edge = torch.tensor(0.0, device=device)
        if edge_logits is not None and edge_targets is not None:
            if edge_targets.dtype != torch.float32:
                edge_targets = edge_targets.float()
            if edge_targets.dim() == 3:
                edge_targets = edge_targets.unsqueeze(1)
            # ensure same spatial size
            if edge_targets.shape[-2:] != edge_logits.shape[-2:]:
                edge_targets = torch.nn.functional.interpolate(edge_targets, size=edge_logits.shape[-2:], mode='nearest')
            loss_edge = self.edge_criterion(edge_logits, edge_targets.to(device))

        s1, s2 = self.log_vars[0], self.log_vars[1]
        total = 0.5 * torch.exp(-s1) * loss_seg + 0.5 * s1
        total = total + 0.5 * torch.exp(-s2) * loss_edge + 0.5 * s2

        return total, {"seg_loss": loss_seg.detach(), "edge_loss": loss_edge.detach(), "log_vars": self.log_vars.detach().cpu()}
