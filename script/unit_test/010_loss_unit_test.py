# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import pytest

from modules.head.loss.base_loss.focal_loss import FocalLoss


def test_FOCALLOSSCPUVSGPU():
    def loss_ce(pred, target):
        loss = -(pred.log() * target + (1 - pred).log() * (1 - target))
        return loss

    def sigmoid_focal_loss_cpu(
        pred, target, num_classes, avg_factor: int, alpha=0.25, gamma=2.0
    ):
        target = torch.nn.functional.one_hot(target, num_classes=num_classes)  # (2, 3)
        pred_sigmoid = pred.sigmoid()  # (2, 3)
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred, target.float(), reduction="none"
        )
        ce_loss2 = loss_ce(pred_sigmoid, target)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
        focal_loss = focal_weight * ce_loss
        eps = torch.finfo(torch.float32).eps
        loss = focal_loss.sum() / (avg_factor + eps)

        gap = torch.abs(ce_loss - ce_loss2) < 1e-2
        assert (gap).all().item()
        return loss

    nums_pos = 1
    num_anchors = 2
    num_classes = 3
    pred = (
        torch.tensor([[5.4742, -0.43376, -4.6261], [4.9875, -4.8535, -5.2556]])
        .float()
        .reshape(num_anchors, num_classes)
    )
    target = torch.tensor([0, 2]).reshape(num_anchors)
    focal_loss_cpu = sigmoid_focal_loss_cpu(
        pred, target, num_classes, avg_factor=nums_pos
    )
    focal_loss_gpu = FocalLoss()(
        pred.clone().cuda(), target.clone().cuda(), avg_factor=nums_pos
    )
    gap = torch.abs(focal_loss_cpu - focal_loss_gpu.cpu()) < 1e-2
    assert (gap).all().item()

    pass


if __name__ == "__main__":
    # [shell] pytest -W ignore::DeprecationWarning -s file
    pytest.main(["-s", "script/unit_test/010_loss_unit_test.py"])
    # test_FOCALLOSSCPUVSGPU()
