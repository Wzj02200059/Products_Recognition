import torch
from torch import nn, Tensor


class rzloss(nn.Module):
    def __init__(self, margin: float, gamma: float) -> None:
        super(rzloss, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, feat: Tensor, target: Tensor) -> Tensor:
        feat_vec = torch.clamp_min(feat + self.margin, min=0).detach()
        target_feat_values = - feat.gather(dim=1, index=target.unsqueeze(1))
        # Sp = target_feat_values + margin + bias
        target_feat_values = target_feat_values + 1 + self.margin
        sp = torch.clamp_min(target_feat_values, min=0, ).detach()
        # feat trans to Sn + Sp
        feat_vec.scatter_(dim=1, index=target.unsqueeze(1), src=sp)

        sigma = torch.ones_like(feat, device=feat.device, dtype=feat.dtype) * self.margin
        src = torch.ones_like(target.unsqueeze(1), dtype=feat.dtype, device=feat.device) - self.margin
        sigma.scatter_(dim=1, index=target.unsqueeze(1), src=src)

        fin_vec = feat_vec * (feat - sigma) * self.gamma

        loss = self.cross_entropy(fin_vec, target)
        return loss


if __name__ == '__main__':
    cre = rzloss(margin=0.2, gamma=80)
    feat = torch.tensor([[1.6355e-01, 5.4747e-02, -1.1524e-01, -9.6338e-02, -3.3949e-02,
                          5.3907e-02, -4.2470e-02, -2.3941e-01, -2.7540e-02, -2.0847e-02,
                          -2.9867e-01, -1.9194e-01, -3.8590e-02, 2.7889e-02, 8.7805e-02,
                          8.6446e-01]])
    target = torch.tensor([15])

    loss = cre(feat, target)
    print(loss)